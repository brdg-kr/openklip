import { createHash, randomUUID } from "node:crypto";
import { lookup } from "node:dns/promises";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { isIP } from "node:net";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";
import { runJsonWorker } from "./json-worker.ts";
import {
  FACE_DETECTOR_MODEL,
  FACE_RECOGNIZER_MODEL,
  ensureFaceModels,
  type FaceModelPaths,
} from "./media-models.ts";
import { memberProfileDir, memberProfilesRoot } from "./paths.ts";
import { faceIndexWorkerPath } from "./script-paths.ts";

const MIN_REFERENCE_IMAGES = 3;
const MAX_REFERENCE_IMAGES = 40;
const MAX_REFERENCE_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_REFERENCE_REDIRECTS = 5;
const IMAGE_CONTENT_TYPES = new Map([
  ["image/jpeg", ".jpg"],
  ["image/png", ".png"],
  ["image/webp", ".webp"],
]);

export interface MemberProfileReference {
  confidence: number;
  filename: string;
  sourceSha256: string;
  vectorB64: string;
}

export interface MemberProfile {
  calibration: {
    hardNegativeCount: number;
    status: "hard-negatives-present" | "incomplete";
    thresholdSource: "default-unmeasured";
  };
  createdAt: string;
  displayName: string;
  fingerprint: string;
  groupId?: string;
  id: string;
  model: {
    detector: string;
    recognizer: string;
    vectorDim: number;
  };
  negativeReferences: MemberProfileReference[];
  references: MemberProfileReference[];
  version: 1;
}

export interface CreateMemberProfileInput {
  displayName: string;
  groupId?: string;
  id: string;
  negativeReferenceImagePaths?: string[];
  referenceImagePaths: string[];
}

export interface CreateMemberProfileSourcesInput
  extends Omit<CreateMemberProfileInput, "referenceImagePaths"> {
  referenceImagePaths?: string[];
  referenceImageUrls?: string[];
  negativeReferenceImagePaths?: string[];
  negativeReferenceImageUrls?: string[];
}

export interface FaceWorkerInputImage {
  atSec?: number;
  id: string;
  path: string;
}

export interface FaceWorkerFace {
  box: { height: number; width: number; x: number; y: number };
  confidence: number;
  landmarks: number[][];
  sharpness: number;
  vectorB64: string;
}

export interface FaceWorkerImageResult {
  atSec: number | null;
  faces: FaceWorkerFace[];
  height: number;
  id: string;
  path: string;
  width: number;
}

export interface FaceWorkerResult {
  detectorModel: string;
  images: FaceWorkerImageResult[];
  recognizerModel: string;
  version: 1;
}

export interface RunFaceWorkerOptions {
  modelPaths?: FaceModelPaths;
  signal?: AbortSignal;
  temporaryDirectory: string;
}

function profilePath(memberProfileId: string): string {
  return join(memberProfileDir(memberProfileId), "profile.json");
}

function profileFingerprintPayload(
  profile: Omit<MemberProfile, "createdAt" | "fingerprint">
): string {
  return JSON.stringify({
    version: profile.version,
    id: profile.id,
    groupId: profile.groupId ?? null,
    model: profile.model,
    references: profile.references.map((reference) => ({
      sourceSha256: reference.sourceSha256,
      vectorB64: reference.vectorB64,
    })),
    negativeReferences: profile.negativeReferences.map((reference) => ({
      sourceSha256: reference.sourceSha256,
      vectorB64: reference.vectorB64,
    })),
  });
}

export function memberProfileFingerprint(
  profile: Omit<MemberProfile, "fingerprint">
): string {
  return createHash("sha256")
    .update(
      profileFingerprintPayload(
        profile as Omit<MemberProfile, "createdAt" | "fingerprint">
      )
    )
    .digest("hex");
}

function normalizeMemberProfile(raw: MemberProfile): MemberProfile {
  const negativeReferences = raw.negativeReferences ?? [];
  const calibration = raw.calibration ?? {
    hardNegativeCount: negativeReferences.length,
    status:
      negativeReferences.length > 0 ? "hard-negatives-present" : "incomplete",
    thresholdSource: "default-unmeasured",
  };
  calibration.status =
    (calibration.status as string) === "calibrated"
      ? "hard-negatives-present"
      : calibration.status;
  calibration.thresholdSource ??= "default-unmeasured";
  const withoutFingerprint = {
    ...raw,
    negativeReferences,
    calibration,
  };
  return {
    ...withoutFingerprint,
    fingerprint:
      raw.fingerprint ?? memberProfileFingerprint(withoutFingerprint),
  };
}

function assertProfileInput(input: CreateMemberProfileInput): void {
  if (!input.id.trim()) {
    throw new Error("member profile id is required");
  }
  if (!input.displayName.trim()) {
    throw new Error("member display name is required");
  }
  if (
    input.referenceImagePaths.length < MIN_REFERENCE_IMAGES ||
    input.referenceImagePaths.length > MAX_REFERENCE_IMAGES
  ) {
    throw new Error(
      `member profile requires ${MIN_REFERENCE_IMAGES}-${MAX_REFERENCE_IMAGES} reference images`
    );
  }
  const negativePaths = input.negativeReferenceImagePaths ?? [];
  if (negativePaths.length > MAX_REFERENCE_IMAGES) {
    throw new Error(
      `member profile accepts at most ${MAX_REFERENCE_IMAGES} hard-negative images`
    );
  }
  for (const path of [...input.referenceImagePaths, ...negativePaths]) {
    if (!existsSync(path)) {
      throw new Error(`member reference image does not exist: ${path}`);
    }
  }
}

async function sha256File(path: string): Promise<string> {
  const bytes = await readFile(path);
  return createHash("sha256").update(bytes).digest("hex");
}

function bestReferenceFace(faces: FaceWorkerFace[]): FaceWorkerFace | null {
  const usable = faces.filter(
    (face) =>
      Math.min(face.box.width, face.box.height) >= 48 && face.sharpness >= 20
  );
  if (usable.length === 0) {
    return null;
  }
  return [...usable].sort((left, right) => {
    const leftArea = left.box.width * left.box.height;
    const rightArea = right.box.width * right.box.height;
    return (
      right.confidence * rightArea - left.confidence * leftArea ||
      right.confidence - left.confidence
    );
  })[0];
}

async function sealProfileReferences(
  images: FaceWorkerImageResult[],
  staging: string,
  prefix: "negative" | "reference"
): Promise<MemberProfileReference[]> {
  const references: MemberProfileReference[] = [];
  for (const [index, image] of images.entries()) {
    const face = bestReferenceFace(image.faces);
    if (!face) {
      throw new Error(`no usable face found in ${prefix} image: ${image.path}`);
    }
    const extension = extname(image.path).toLowerCase() || ".jpg";
    const filename = `${prefix}-${String(index + 1).padStart(2, "0")}${extension}`;
    await copyFile(image.path, join(staging, filename));
    references.push({
      confidence: face.confidence,
      filename,
      sourceSha256: await sha256File(image.path),
      vectorB64: face.vectorB64,
    });
  }
  return references;
}

export async function runFaceWorker(
  images: FaceWorkerInputImage[],
  options: RunFaceWorkerOptions
): Promise<FaceWorkerResult> {
  const models = options.modelPaths ?? (await ensureFaceModels());
  return runJsonWorker<
    {
      detectorModel: string;
      images: FaceWorkerInputImage[];
      models: FaceModelPaths;
      recognizerModel: string;
    },
    FaceWorkerResult
  >({
    label: "face-index",
    signal: options.signal,
    temporaryDirectory: options.temporaryDirectory,
    job: {
      detectorModel: FACE_DETECTOR_MODEL.id,
      images,
      models,
      recognizerModel: FACE_RECOGNIZER_MODEL.id,
    },
    args: (jobPath, outputPath) => [faceIndexWorkerPath(), jobPath, outputPath],
  });
}

export function loadMemberProfile(memberProfileId: string): MemberProfile {
  const path = profilePath(memberProfileId);
  if (!existsSync(path)) {
    throw new Error(`member profile not found: ${memberProfileId}`);
  }
  const parsed = normalizeMemberProfile(
    JSON.parse(readFileSync(path, "utf8")) as MemberProfile
  );
  if (parsed.id !== memberProfileId || parsed.version !== 1) {
    throw new Error(`member profile is invalid: ${memberProfileId}`);
  }
  return parsed;
}

export function listMemberProfiles(): MemberProfile[] {
  const root = memberProfilesRoot();
  if (!existsSync(root)) {
    return [];
  }
  const profiles: MemberProfile[] = [];
  for (const directory of readdirSync(root)) {
    const path = join(root, directory, "profile.json");
    if (!existsSync(path)) {
      continue;
    }
    try {
      profiles.push(
        normalizeMemberProfile(
          JSON.parse(readFileSync(path, "utf8")) as MemberProfile
        )
      );
    } catch {
      // One corrupt profile does not hide other valid profiles.
    }
  }
  return profiles.sort(
    (left, right) =>
      left.displayName.localeCompare(right.displayName) ||
      left.id.localeCompare(right.id)
  );
}

function isPrivateAddress(address: string): boolean {
  const mappedIpv4 = address
    .toLowerCase()
    .match(/^::ffff:(\d+\.\d+\.\d+\.\d+)$/);
  if (mappedIpv4) {
    return isPrivateAddress(mappedIpv4[1]);
  }
  if (isIP(address) === 4) {
    const parts = address.split(".").map(Number);
    return (
      parts[0] === 0 ||
      parts[0] === 10 ||
      parts[0] === 127 ||
      (parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127) ||
      (parts[0] === 169 && parts[1] === 254) ||
      (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
      (parts[0] === 192 && parts[1] === 168) ||
      parts[0] >= 224
    );
  }
  const normalized = address.toLowerCase();
  return (
    normalized === "::" ||
    normalized === "::1" ||
    normalized.startsWith("fc") ||
    normalized.startsWith("fd") ||
    normalized.startsWith("fe8") ||
    normalized.startsWith("fe9") ||
    normalized.startsWith("fea") ||
    normalized.startsWith("feb")
  );
}

async function assertPublicReferenceUrl(raw: string): Promise<URL> {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(`reference image URL is invalid: ${raw}`);
  }
  if (parsed.protocol !== "https:") {
    throw new Error("reference image URLs must use HTTPS");
  }
  if (parsed.username || parsed.password) {
    throw new Error("reference image URLs must not contain credentials");
  }
  const hostname = parsed.hostname.toLowerCase();
  if (hostname === "localhost" || hostname.endsWith(".local")) {
    throw new Error("reference image URL must use a public host");
  }
  const addresses = await lookup(hostname, { all: true });
  if (
    addresses.length === 0 ||
    addresses.some((entry) => isPrivateAddress(entry.address))
  ) {
    throw new Error("reference image URL resolved to a private address");
  }
  return parsed;
}

export async function readResponseBodyLimited(
  response: Response,
  maxBytes: number
): Promise<Uint8Array> {
  if (!response.body) {
    throw new Error("reference image response has no body");
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    totalBytes += value.byteLength;
    if (totalBytes > maxBytes) {
      await reader.cancel();
      throw new Error("reference image exceeds the 20 MB limit");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function downloadReferenceImage(
  rawUrl: string,
  directory: string,
  index: number,
  label: "negative" | "reference",
  fetchImpl: typeof fetch
): Promise<string> {
  let current = await assertPublicReferenceUrl(rawUrl);
  for (let redirect = 0; redirect <= MAX_REFERENCE_REDIRECTS; redirect += 1) {
    const response = await fetchImpl(current, { redirect: "manual" });
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (!location || redirect === MAX_REFERENCE_REDIRECTS) {
        throw new Error(`reference image redirect failed: ${rawUrl}`);
      }
      current = await assertPublicReferenceUrl(
        new URL(location, current).toString()
      );
      continue;
    }
    if (!response.ok) {
      throw new Error(
        `reference image download failed with HTTP ${response.status}`
      );
    }
    const contentType = response.headers.get("content-type")?.split(";")[0];
    const extension = contentType ? IMAGE_CONTENT_TYPES.get(contentType) : null;
    if (!extension) {
      throw new Error(
        `reference URL did not return a supported image: ${rawUrl}`
      );
    }
    const declaredLength = Number(response.headers.get("content-length") ?? 0);
    if (declaredLength > MAX_REFERENCE_IMAGE_BYTES) {
      throw new Error("reference image exceeds the 20 MB limit");
    }
    const bytes = await readResponseBodyLimited(
      response,
      MAX_REFERENCE_IMAGE_BYTES
    );
    const path = join(
      directory,
      `remote-${label}-${String(index + 1).padStart(2, "0")}${extension}`
    );
    await writeFile(path, bytes);
    return path;
  }
  throw new Error(`reference image redirect limit exceeded: ${rawUrl}`);
}

export async function createMemberProfileFromSources(
  input: CreateMemberProfileSourcesInput,
  options: {
    fetchImpl?: typeof fetch;
    force?: boolean;
    modelPaths?: FaceModelPaths;
    runWorker?: typeof runFaceWorker;
    signal?: AbortSignal;
  } = {}
): Promise<MemberProfile> {
  const localPaths = input.referenceImagePaths ?? [];
  const urls = input.referenceImageUrls ?? [];
  const negativeLocalPaths = input.negativeReferenceImagePaths ?? [];
  const negativeUrls = input.negativeReferenceImageUrls ?? [];
  const total = localPaths.length + urls.length;
  if (total < MIN_REFERENCE_IMAGES || total > MAX_REFERENCE_IMAGES) {
    throw new Error(
      `member profile requires ${MIN_REFERENCE_IMAGES}-${MAX_REFERENCE_IMAGES} reference images`
    );
  }
  if (negativeLocalPaths.length + negativeUrls.length > MAX_REFERENCE_IMAGES) {
    throw new Error(
      `member profile accepts at most ${MAX_REFERENCE_IMAGES} hard-negative images`
    );
  }
  if (urls.length === 0 && negativeUrls.length === 0) {
    return createMemberProfile(
      {
        id: input.id,
        displayName: input.displayName,
        groupId: input.groupId,
        referenceImagePaths: localPaths,
        negativeReferenceImagePaths: negativeLocalPaths,
      },
      options
    );
  }
  const temporary = await mkdtemp(join(tmpdir(), "openklip-member-refs-"));
  try {
    const downloaded: string[] = [];
    for (const [index, url] of urls.entries()) {
      downloaded.push(
        await downloadReferenceImage(
          url,
          temporary,
          index,
          "reference",
          options.fetchImpl ?? fetch
        )
      );
    }
    const downloadedNegatives: string[] = [];
    for (const [index, url] of negativeUrls.entries()) {
      downloadedNegatives.push(
        await downloadReferenceImage(
          url,
          temporary,
          index,
          "negative",
          options.fetchImpl ?? fetch
        )
      );
    }
    return await createMemberProfile(
      {
        id: input.id,
        displayName: input.displayName,
        groupId: input.groupId,
        referenceImagePaths: [...localPaths, ...downloaded],
        negativeReferenceImagePaths: [
          ...negativeLocalPaths,
          ...downloadedNegatives,
        ],
      },
      options
    );
  } finally {
    await rm(temporary, { force: true, recursive: true });
  }
}

export async function createMemberProfile(
  input: CreateMemberProfileInput,
  options: {
    force?: boolean;
    modelPaths?: FaceModelPaths;
    runWorker?: typeof runFaceWorker;
    signal?: AbortSignal;
  } = {}
): Promise<MemberProfile> {
  assertProfileInput(input);
  const directory = memberProfileDir(input.id);
  const finalPath = profilePath(input.id);
  if (existsSync(finalPath) && !options.force) {
    throw new Error(
      `member profile already exists: ${input.id} (set force to replace)`
    );
  }
  const staging = `${directory}.${process.pid}.${randomUUID()}.staging`;
  await mkdir(staging, { recursive: true });
  try {
    const negativePaths = input.negativeReferenceImagePaths ?? [];
    const positiveCount = input.referenceImagePaths.length;
    const workerResult = await (options.runWorker ?? runFaceWorker)(
      [
        ...input.referenceImagePaths.map((path, index) => ({
          id: `reference-${index + 1}`,
          path,
        })),
        ...negativePaths.map((path, index) => ({
          id: `negative-${index + 1}`,
          path,
        })),
      ],
      {
        modelPaths: options.modelPaths,
        signal: options.signal,
        temporaryDirectory: staging,
      }
    );
    const references = await sealProfileReferences(
      workerResult.images.slice(0, positiveCount),
      staging,
      "reference"
    );
    const negativeReferences = await sealProfileReferences(
      workerResult.images.slice(positiveCount),
      staging,
      "negative"
    );
    if (
      new Set(references.map((reference) => reference.sourceSha256)).size < 3
    ) {
      throw new Error("member profile requires at least three distinct images");
    }
    const profileWithoutFingerprint: Omit<MemberProfile, "fingerprint"> = {
      version: 1,
      id: input.id.trim(),
      displayName: input.displayName.trim(),
      ...(input.groupId?.trim() ? { groupId: input.groupId.trim() } : {}),
      createdAt: new Date().toISOString(),
      model: {
        detector: workerResult.detectorModel,
        recognizer: workerResult.recognizerModel,
        vectorDim: 128,
      },
      references,
      negativeReferences,
      calibration: {
        hardNegativeCount: negativeReferences.length,
        status:
          negativeReferences.length > 0
            ? "hard-negatives-present"
            : "incomplete",
        thresholdSource: "default-unmeasured",
      },
    };
    const profile: MemberProfile = {
      ...profileWithoutFingerprint,
      fingerprint: memberProfileFingerprint(profileWithoutFingerprint),
    };
    await writeFile(
      join(staging, "profile.json"),
      JSON.stringify(profile, null, 2)
    );
    await rm(directory, { force: true, recursive: true });
    await mkdir(memberProfilesRoot(), { recursive: true });
    await rename(staging, directory);
    return profile;
  } catch (error) {
    await rm(staging, { force: true, recursive: true });
    throw error;
  }
}
