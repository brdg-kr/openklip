import { createHash, randomUUID } from "node:crypto";
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import sharp from "sharp";
import { ProcessCancelledError } from "./ffmpeg.ts";
import { runJsonWorker } from "./json-worker.ts";
import {
  extractFrames,
  extractRefinementFrames,
  extractSceneCutFrames,
  refinementWindows,
  replaceHitsInWindows,
  type TimedFrame,
} from "./media-index-frames.ts";
import {
  buildMemberAppearanceTracks,
  bytesToVector,
  DEFAULT_FACE_MATCH_OPTIONS,
  type IndexedFaceHit,
  matchFaceVector,
  type MemberAppearanceTrack,
  type ReferenceVector,
  vectorToBytes,
} from "./media-index-core.ts";
import {
  FACE_DETECTOR_MODEL,
  FACE_RECOGNIZER_MODEL,
  SCENE_EMBEDDING_MODEL,
} from "./media-models.ts";
import {
  type MediaSearchResult,
  rankMediaScenes,
} from "./media-index-search.ts";
import {
  type MediaIndexFile,
  mediaIndexFreshnessError,
  readMediaIndexFile,
} from "./media-index-storage.ts";
import {
  listMemberProfiles,
  loadMemberProfile,
  runFaceWorker,
  type FaceWorkerImageResult,
} from "./member-profiles.ts";
import { projectPaths } from "./paths.ts";
import { loadProject } from "./projectStore.ts";
import { mediaSceneEmbedScriptPath } from "./script-paths.ts";

const FACE_FPS = 4;
const FACE_REFINE_FPS = 10;
const FACE_FRAME_STEP_SEC = 1 / FACE_FPS;
const SCENE_FPS = 1;
const SCENE_FRAME_STEP_SEC = 1 / SCENE_FPS;
const MIN_FACE_SIZE_PX = 32;
const MIN_FACE_SHARPNESS = 8;

export type MediaIndexPhase =
  | "face-frames"
  | "face-index"
  | "face-refine"
  | "scene-frames"
  | "scene-index"
  | "store";

export type MediaIndexStatusValue =
  | "cancelled"
  | "done"
  | "error"
  | "interrupted"
  | "running";

export interface MediaIndexStatus {
  createdAt: string;
  error?: string;
  faceDetections?: number;
  faceTracks?: number;
  memberProfileId?: string;
  phase?: MediaIndexPhase;
  sceneEmbeddings?: number;
  slug: string;
  sourceSha256?: string;
  status: MediaIndexStatusValue;
  updatedAt: string;
  version: 1;
}

export interface BuildMediaIndexInput {
  memberProfileId?: string;
  onProgress?: (status: MediaIndexStatus) => void;
  signal?: AbortSignal;
  slug: string;
}

export interface StartMediaIndexResult {
  slug: string;
  status: MediaIndexStatusValue;
}

interface SceneInput {
  atSec: number;
  id: string;
  memberProfileId?: string;
  path: string;
  scope: "full" | "target-context" | "target-medium";
}

interface SceneWorkerResult {
  dim: number;
  dtype: string;
  images: Array<SceneInput & { vectorB64: string }>;
  model: string;
  version: 1;
}

const liveBuilds = new Map<
  string,
  { controller: AbortController; promise: Promise<void> }
>();

function now(): string {
  return new Date().toISOString();
}

async function atomicWriteJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, path);
}

async function persistStatus(status: MediaIndexStatus): Promise<void> {
  await atomicWriteJson(projectPaths(status.slug).mediaIndexStatus, status);
}

function statusUpdate(
  current: MediaIndexStatus,
  patch: Partial<MediaIndexStatus>,
  onProgress?: (status: MediaIndexStatus) => void
): MediaIndexStatus {
  const updated = { ...current, ...patch, updatedAt: now() };
  onProgress?.(updated);
  return updated;
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return hash.digest("hex");
}

function decodeVector(vectorB64: string): Float32Array {
  return bytesToVector(Buffer.from(vectorB64, "base64"));
}

function profileReferences(memberProfileId: string): ReferenceVector[] {
  const profile = loadMemberProfile(memberProfileId);
  return profile.references.map((reference) => ({
    memberProfileId,
    vector: decodeVector(reference.vectorB64),
  }));
}

function negativeProfileReferences(memberProfileId: string): ReferenceVector[] {
  const target = loadMemberProfile(memberProfileId);
  const references: ReferenceVector[] = target.negativeReferences.map(
    (reference) => ({
      memberProfileId: `${target.id}:hard-negative`,
      vector: decodeVector(reference.vectorB64),
    })
  );
  if (!target.groupId) {
    return references;
  }
  for (const profile of listMemberProfiles()) {
    if (profile.id === target.id || profile.groupId !== target.groupId) {
      continue;
    }
    for (const reference of profile.references) {
      references.push({
        memberProfileId: profile.id,
        vector: decodeVector(reference.vectorB64),
      });
    }
  }
  return references;
}

function indexedFaceHits(
  images: FaceWorkerImageResult[],
  memberProfileId?: string
): IndexedFaceHit[] {
  const targetReferences = memberProfileId
    ? profileReferences(memberProfileId)
    : [];
  const negativeReferences = memberProfileId
    ? negativeProfileReferences(memberProfileId)
    : [];
  const hits: IndexedFaceHit[] = [];
  for (const image of images) {
    if (image.atSec === null) {
      continue;
    }
    for (const face of image.faces) {
      const vector = decodeVector(face.vectorB64);
      const qualityPassed =
        Math.min(face.box.width, face.box.height) >= MIN_FACE_SIZE_PX &&
        face.sharpness >= MIN_FACE_SHARPNESS;
      const match =
        memberProfileId && qualityPassed
          ? matchFaceVector(
              vector,
              targetReferences,
              negativeReferences,
              DEFAULT_FACE_MATCH_OPTIONS
            )
          : {
              margin: 0,
              memberProfileId: null,
              score: 0,
              status: "unknown" as const,
            };
      hits.push({
        atSec: image.atSec,
        box: face.box,
        confidence: face.confidence,
        vector,
        ...match,
      });
    }
  }
  return hits;
}

function nearestMatchedFace(
  hits: IndexedFaceHit[],
  atSec: number,
  memberProfileId: string
): IndexedFaceHit | null {
  let nearest: IndexedFaceHit | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const hit of hits) {
    if (hit.memberProfileId !== memberProfileId || hit.status !== "matched") {
      continue;
    }
    const distance = Math.abs(hit.atSec - atSec);
    if (distance <= FACE_FRAME_STEP_SEC && distance < nearestDistance) {
      nearest = hit;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function cropBox(
  width: number,
  height: number,
  face: IndexedFaceHit,
  scale: number
): { height: number; left: number; top: number; width: number } {
  const centerX = face.box.x + face.box.width / 2;
  const centerY = face.box.y + face.box.height * 1.6;
  const cropWidth = Math.min(width, Math.max(face.box.width * scale, 160));
  const cropHeight = Math.min(
    height,
    Math.max(face.box.height * scale * 1.2, 200)
  );
  const left = Math.round(
    Math.max(0, Math.min(width - cropWidth, centerX - cropWidth / 2))
  );
  const top = Math.round(
    Math.max(0, Math.min(height - cropHeight, centerY - cropHeight / 2))
  );
  return {
    left,
    top,
    width: Math.max(1, Math.round(cropWidth)),
    height: Math.max(1, Math.round(cropHeight)),
  };
}

async function sceneInputs(
  frames: TimedFrame[],
  hits: IndexedFaceHit[],
  memberProfileId?: string
): Promise<SceneInput[]> {
  const inputs: SceneInput[] = [];
  for (const [index, frame] of frames.entries()) {
    const { atSec, path } = frame;
    const face = memberProfileId
      ? nearestMatchedFace(hits, atSec, memberProfileId)
      : null;
    inputs.push({
      atSec,
      id: `scene-${index + 1}-full`,
      path,
      scope: "full",
      ...(face && memberProfileId ? { memberProfileId } : {}),
    });
    if (!(memberProfileId && face)) {
      continue;
    }
    const metadata = await sharp(path).metadata();
    if (!(metadata.width && metadata.height)) {
      continue;
    }
    for (const crop of [
      { scope: "target-medium" as const, scale: 4 },
      { scope: "target-context" as const, scale: 7 },
    ]) {
      const outputPath = join(
        dirname(path),
        `${basename(path, ".jpg")}-${crop.scope}.jpg`
      );
      await sharp(path)
        .extract(cropBox(metadata.width, metadata.height, face, crop.scale))
        .jpeg({ quality: 82 })
        .toFile(outputPath);
      inputs.push({
        atSec,
        id: `scene-${index + 1}-${crop.scope}`,
        memberProfileId,
        path: outputPath,
        scope: crop.scope,
      });
    }
  }
  return inputs;
}

function runSceneWorker(
  inputs: SceneInput[],
  directory: string,
  signal?: AbortSignal
): Promise<SceneWorkerResult> {
  return runJsonWorker<{ images: SceneInput[] }, SceneWorkerResult>({
    label: "media-scene",
    signal,
    temporaryDirectory: directory,
    job: { images: inputs },
    args: (jobPath, outputPath) => [
      mediaSceneEmbedScriptPath(),
      "index",
      jobPath,
      outputPath,
      SCENE_EMBEDDING_MODEL.id,
      SCENE_EMBEDDING_MODEL.dtype,
    ],
  });
}

async function writeIndexFile(
  path: string,
  metadata: Record<string, string>,
  hits: IndexedFaceHit[],
  tracks: MemberAppearanceTrack[],
  scenes: SceneWorkerResult
): Promise<void> {
  const index: MediaIndexFile = {
    version: 1,
    metadata,
    faceHits: hits.map((hit) => ({
      atSec: hit.atSec,
      box: hit.box,
      confidence: hit.confidence,
      memberProfileId: hit.memberProfileId,
      status: hit.status,
      score: hit.score,
      margin: hit.margin,
      vectorB64: Buffer.from(vectorToBytes(hit.vector)).toString("base64"),
    })),
    faceTracks: tracks,
    scenes: scenes.images.map((scene) => ({
      id: scene.id,
      atSec: scene.atSec,
      fromSec: scene.atSec,
      toSec: scene.atSec + SCENE_FRAME_STEP_SEC,
      scope: scene.scope,
      ...(scene.memberProfileId
        ? { memberProfileId: scene.memberProfileId }
        : {}),
      thumbnail: scene.path,
      vectorB64: scene.vectorB64,
    })),
  };
  await atomicWriteJson(path, index);
}

export async function buildMediaIndex({
  memberProfileId,
  onProgress,
  signal,
  slug,
}: BuildMediaIndexInput): Promise<MediaIndexStatus> {
  const paths = projectPaths(slug);
  const project = await loadProject(slug);
  if (!existsSync(project.source)) {
    throw new Error(`project source does not exist: ${project.source}`);
  }
  const memberProfile = memberProfileId
    ? loadMemberProfile(memberProfileId)
    : null;
  let status: MediaIndexStatus = {
    version: 1,
    slug,
    status: "running",
    ...(memberProfileId ? { memberProfileId } : {}),
    createdAt: now(),
    updatedAt: now(),
  };
  await persistStatus(status);
  const update = async (patch: Partial<MediaIndexStatus>) => {
    status = statusUpdate(status, patch, onProgress);
    await persistStatus(status);
  };
  try {
    const sourceSha256 = await sha256File(project.source);
    const sourceStat = statSync(project.source);
    await update({ phase: "face-frames", sourceSha256 });
    const faceFrames = await extractFrames(
      paths.proxy,
      paths.mediaFaceFrames,
      FACE_FPS,
      signal
    );
    await update({ phase: "face-index" });
    const faceResult = await runFaceWorker(
      faceFrames.map((path, index) => ({
        id: basename(path, ".jpg"),
        path,
        atSec: index * FACE_FRAME_STEP_SEC,
      })),
      { signal, temporaryDirectory: paths.working }
    );
    const coarseHits = indexedFaceHits(faceResult.images, memberProfileId);
    await update({ phase: "scene-frames" });
    const fixedScenePaths = await extractFrames(
      paths.proxy,
      paths.mediaSceneFrames,
      SCENE_FPS,
      signal
    );
    const fixedSceneFrames = fixedScenePaths.map((path, index) => ({
      path,
      atSec: index * SCENE_FRAME_STEP_SEC,
    }));
    const cutSceneFrames = await extractSceneCutFrames(
      paths.proxy,
      paths.mediaSceneFrames,
      signal
    );
    const durationSec = project.durationSamples / project.sampleRate;
    const windows = memberProfileId
      ? refinementWindows(
          coarseHits,
          cutSceneFrames.map((frame) => frame.atSec),
          durationSec
        )
      : [];
    let hits = coarseHits;
    if (windows.length > 0) {
      await update({ phase: "face-refine" });
      const refinementFrames = await extractRefinementFrames(
        paths.proxy,
        paths.mediaFaceFrames,
        windows,
        signal
      );
      const refinedResult = await runFaceWorker(
        refinementFrames.map((frame, index) => ({
          id: `refined-${index + 1}`,
          path: frame.path,
          atSec: frame.atSec,
        })),
        { signal, temporaryDirectory: paths.working }
      );
      hits = replaceHitsInWindows(
        coarseHits,
        indexedFaceHits(refinedResult.images, memberProfileId),
        windows
      );
    }
    const tracks = buildMemberAppearanceTracks(hits, {
      frameStepSec: 1 / FACE_REFINE_FPS,
      maxGapSec: 0.65,
    });
    await rm(paths.mediaFaceFrames, { force: true, recursive: true });
    await update({
      phase: "scene-index",
      faceDetections: hits.length,
      faceTracks: tracks.length,
    });
    const sceneFrames = [...fixedSceneFrames, ...cutSceneFrames].sort(
      (left, right) => left.atSec - right.atSec
    );
    const inputs = await sceneInputs(sceneFrames, hits, memberProfileId);
    const sceneResult = await runSceneWorker(inputs, paths.working, signal);
    await update({
      phase: "store",
      sceneEmbeddings: sceneResult.images.length,
    });
    await writeIndexFile(
      paths.mediaIndex,
      {
        version: "1",
        sourceSha256,
        sourcePath: project.source,
        sourceSize: String(sourceStat.size),
        sourceMtimeMs: String(sourceStat.mtimeMs),
        faceDetectorModel: FACE_DETECTOR_MODEL.id,
        faceRecognizerModel: FACE_RECOGNIZER_MODEL.id,
        sceneModel: sceneResult.model,
        sceneDtype: sceneResult.dtype,
        sceneDim: String(sceneResult.dim),
        memberProfileId: memberProfileId ?? "",
        memberProfileFingerprint: memberProfile?.fingerprint ?? "",
      },
      hits,
      tracks,
      sceneResult
    );
    await update({ status: "done" });
    return status;
  } catch (error) {
    const cancelled = signal?.aborted || error instanceof ProcessCancelledError;
    await update({
      status: cancelled ? "cancelled" : "error",
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  } finally {
    await rm(paths.mediaFaceFrames, { force: true, recursive: true });
  }
}

export function startMediaIndexBuild(
  input: Omit<BuildMediaIndexInput, "signal">
): StartMediaIndexResult {
  if (liveBuilds.has(input.slug)) {
    throw new Error(`media index build already running for ${input.slug}`);
  }
  const controller = new AbortController();
  const promise = buildMediaIndex({ ...input, signal: controller.signal })
    .then(() => undefined)
    .catch(() => undefined)
    .finally(() => liveBuilds.delete(input.slug));
  liveBuilds.set(input.slug, { controller, promise });
  return { slug: input.slug, status: "running" };
}

export function cancelMediaIndexBuild(slug: string): boolean {
  const live = liveBuilds.get(slug);
  if (!live) {
    return false;
  }
  live.controller.abort();
  return true;
}

export function readMediaIndexStatus(slug: string): MediaIndexStatus | null {
  const path = projectPaths(slug).mediaIndexStatus;
  if (!existsSync(path)) {
    return null;
  }
  const status = JSON.parse(readFileSync(path, "utf8")) as MediaIndexStatus;
  const index = readMediaIndexFile(projectPaths(slug).mediaIndex);
  if (status.status === "done" && !index) {
    return {
      ...status,
      status: "interrupted",
      error: "media index files are missing; rebuild required",
      updatedAt: now(),
    };
  }
  if (status.status === "done" && index) {
    const freshnessError = mediaIndexFreshnessError(slug, index);
    if (freshnessError) {
      return {
        ...status,
        status: "interrupted",
        error: freshnessError,
        updatedAt: now(),
      };
    }
  }
  if (status.status === "running" && !liveBuilds.has(slug)) {
    return { ...status, status: "interrupted", updatedAt: now() };
  }
  return status;
}

export function memberAppearanceTracks(
  slug: string,
  memberProfileId: string,
  includeAmbiguous = false
): MemberAppearanceTrack[] {
  if (readMediaIndexStatus(slug)?.status !== "done") {
    return [];
  }
  const path = projectPaths(slug).mediaIndex;
  const index = readMediaIndexFile(path);
  if (!index) {
    return [];
  }
  const rows = index.faceTracks
    .filter((row) => row.memberProfileId === memberProfileId)
    .sort((left, right) => left.startSec - right.startSec);
  return includeAmbiguous
    ? rows
    : rows.filter((row) => row.status === "matched");
}

async function embedSceneQueries(
  queries: string[],
  temporaryDirectory: string
): Promise<Array<{ query: string; vector: Float32Array }>> {
  if (queries.length < 1 || queries.length > 3) {
    throw new Error("media search requires 1-3 queries");
  }
  const jobPath = join(
    temporaryDirectory,
    `scene-query-${process.pid}-${randomUUID()}.json`
  );
  await writeFile(jobPath, JSON.stringify({ queries }));
  const processHandle = Bun.spawn(
    [
      "node",
      mediaSceneEmbedScriptPath(),
      "queries",
      jobPath,
      SCENE_EMBEDDING_MODEL.id,
      SCENE_EMBEDDING_MODEL.dtype,
    ],
    { stdout: "pipe", stderr: "inherit" }
  );
  try {
    const stdout = await new Response(processHandle.stdout).text();
    const exitCode = await processHandle.exited;
    if (exitCode !== 0) {
      throw new Error(
        `scene query embedding failed with exit code ${exitCode}`
      );
    }
    const parsed = JSON.parse(stdout.trim()) as {
      vectors: Array<{ text: string; vector: number[] }>;
    };
    return parsed.vectors.map((item) => ({
      query: item.text,
      vector: Float32Array.from(item.vector),
    }));
  } finally {
    await rm(jobPath, { force: true });
  }
}

export async function searchMediaIndex(
  slug: string,
  query: string | string[],
  options: { limit?: number; memberProfileId?: string } = {}
): Promise<MediaSearchResult[]> {
  if (readMediaIndexStatus(slug)?.status !== "done") {
    return [];
  }
  const path = projectPaths(slug).mediaIndex;
  const index = readMediaIndexFile(path);
  if (!index) {
    return [];
  }
  const queries = Array.from(
    new Set(
      (Array.isArray(query) ? query : [query])
        .map((item) => item.trim())
        .filter(Boolean)
    )
  );
  if (queries.length === 0 || queries.length > 3) {
    throw new Error("scene search requires one to three queries");
  }
  const queryVectors = await embedSceneQueries(
    queries,
    projectPaths(slug).working
  );
  return rankMediaScenes(index.scenes, queryVectors, options);
}
