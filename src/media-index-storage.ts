import { existsSync, readFileSync, statSync } from "node:fs";
import { rm } from "node:fs/promises";
import type {
  IndexedFaceHit,
  MemberAppearanceTrack,
} from "./media-index-core.ts";
import {
  FACE_DETECTOR_MODEL,
  FACE_RECOGNIZER_MODEL,
  SCENE_EMBEDDING_MODEL,
} from "./media-models.ts";
import { loadMemberProfile } from "./member-profiles.ts";
import { projectPaths } from "./paths.ts";

export interface StoredFaceHit {
  atSec: number;
  box: IndexedFaceHit["box"];
  confidence: number;
  margin: number;
  memberProfileId: string | null;
  score: number;
  status: IndexedFaceHit["status"];
  vectorB64: string;
}

export interface StoredSceneEmbedding {
  atSec: number;
  fromSec: number;
  id: string;
  memberProfileId?: string;
  scope: string;
  thumbnail: string;
  toSec: number;
  vectorB64: string;
}

export interface MediaIndexFile {
  faceHits: StoredFaceHit[];
  faceTracks: MemberAppearanceTrack[];
  metadata: Record<string, string>;
  scenes: StoredSceneEmbedding[];
  version: 1;
}

export function readMediaIndexFile(path: string): MediaIndexFile | null {
  if (!existsSync(path)) {
    return null;
  }
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as MediaIndexFile;
    return parsed.version === 1 ? parsed : null;
  } catch {
    return null;
  }
}

function currentProjectSource(slug: string): string | null {
  try {
    const project = JSON.parse(
      readFileSync(projectPaths(slug).project, "utf8")
    ) as { source?: unknown };
    return typeof project.source === "string" ? project.source : null;
  } catch {
    return null;
  }
}

export function mediaIndexFreshnessError(
  slug: string,
  index: MediaIndexFile
): string | null {
  if (
    index.metadata.faceDetectorModel !== FACE_DETECTOR_MODEL.id ||
    index.metadata.faceRecognizerModel !== FACE_RECOGNIZER_MODEL.id ||
    index.metadata.sceneModel !== SCENE_EMBEDDING_MODEL.id ||
    index.metadata.sceneDtype !== SCENE_EMBEDDING_MODEL.dtype
  ) {
    return "media index model contract changed; rebuild required";
  }
  const source = currentProjectSource(slug);
  if (!source || source !== index.metadata.sourcePath || !existsSync(source)) {
    return "project source changed; media index rebuild required";
  }
  const sourceStat = statSync(source);
  if (
    String(sourceStat.size) !== index.metadata.sourceSize ||
    String(sourceStat.mtimeMs) !== index.metadata.sourceMtimeMs
  ) {
    return "project source bytes changed; media index rebuild required";
  }
  const memberProfileId = index.metadata.memberProfileId;
  if (memberProfileId) {
    try {
      const profile = loadMemberProfile(memberProfileId);
      if (profile.fingerprint !== index.metadata.memberProfileFingerprint) {
        return "member profile changed; media index rebuild required";
      }
    } catch {
      return "member profile is unavailable; media index rebuild required";
    }
  }
  return null;
}

export async function invalidateMediaIndex(slug: string): Promise<void> {
  const paths = projectPaths(slug);
  await Promise.all([
    rm(paths.mediaIndex, { force: true }),
    rm(paths.mediaIndexStatus, { force: true }),
    rm(paths.mediaFaceFrames, { force: true, recursive: true }),
    rm(paths.mediaSceneFrames, { force: true, recursive: true }),
  ]);
}
