import assert from "node:assert/strict";
import { existsSync, statSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  invalidateMediaIndex,
  type MediaIndexFile,
  mediaIndexFreshnessError,
} from "../src/media-index-storage.ts";
import {
  FACE_DETECTOR_MODEL,
  FACE_RECOGNIZER_MODEL,
  SCENE_EMBEDDING_MODEL,
} from "../src/media-models.ts";
import { projectPaths } from "../src/paths.ts";

test("media index freshness detects changed source bytes", async () => {
  const root = await mkdtemp(join(tmpdir(), "openklip-index-storage-"));
  const previousRoot = process.env.OPENKLIP_PROJECTS_ROOT;
  process.env.OPENKLIP_PROJECTS_ROOT = root;
  try {
    const slug = "freshness-test";
    const paths = projectPaths(slug);
    await mkdir(paths.working, { recursive: true });
    const source = join(paths.dir, "source.mp4");
    await writeFile(source, "first");
    await writeFile(paths.project, JSON.stringify({ source }));
    const sourceStat = statSync(source);
    const index: MediaIndexFile = {
      version: 1,
      metadata: {
        sourcePath: source,
        sourceSize: String(sourceStat.size),
        sourceMtimeMs: String(sourceStat.mtimeMs),
        faceDetectorModel: FACE_DETECTOR_MODEL.id,
        faceRecognizerModel: FACE_RECOGNIZER_MODEL.id,
        sceneModel: SCENE_EMBEDDING_MODEL.id,
        sceneDtype: SCENE_EMBEDDING_MODEL.dtype,
        memberProfileId: "",
      },
      faceHits: [],
      faceTracks: [],
      scenes: [],
    };
    assert.equal(mediaIndexFreshnessError(slug, index), null);

    await writeFile(source, "changed-source-bytes");
    assert.match(
      mediaIndexFreshnessError(slug, index) ?? "",
      /source bytes changed/
    );
  } finally {
    if (previousRoot === undefined) {
      delete process.env.OPENKLIP_PROJECTS_ROOT;
    } else {
      process.env.OPENKLIP_PROJECTS_ROOT = previousRoot;
    }
    await rm(root, { force: true, recursive: true });
  }
});

test("invalidateMediaIndex removes derived files and frame directories", async () => {
  const root = await mkdtemp(join(tmpdir(), "openklip-index-invalidate-"));
  const previousRoot = process.env.OPENKLIP_PROJECTS_ROOT;
  process.env.OPENKLIP_PROJECTS_ROOT = root;
  try {
    const slug = "invalidate-test";
    const paths = projectPaths(slug);
    await mkdir(paths.mediaFaceFrames, { recursive: true });
    await mkdir(paths.mediaSceneFrames, { recursive: true });
    await writeFile(paths.mediaIndex, "{}");
    await writeFile(paths.mediaIndexStatus, "{}");
    await invalidateMediaIndex(slug);
    assert.equal(existsSync(paths.mediaIndex), false);
    assert.equal(existsSync(paths.mediaIndexStatus), false);
    assert.equal(existsSync(paths.mediaFaceFrames), false);
    assert.equal(existsSync(paths.mediaSceneFrames), false);
  } finally {
    if (previousRoot === undefined) {
      delete process.env.OPENKLIP_PROJECTS_ROOT;
    } else {
      process.env.OPENKLIP_PROJECTS_ROOT = previousRoot;
    }
    await rm(root, { force: true, recursive: true });
  }
});
