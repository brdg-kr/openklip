import assert from "node:assert/strict";
import { existsSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  createMemberProfile,
  createMemberProfileFromSources,
  listMemberProfiles,
  loadMemberProfile,
  readResponseBodyLimited,
  type FaceWorkerResult,
} from "../src/member-profiles.ts";
import { memberProfileDir } from "../src/paths.ts";

test("createMemberProfile seals references and hard-negative calibration", async () => {
  const root = await mkdtemp(join(tmpdir(), "openklip-member-profile-"));
  const previousState = process.env.OPENKLIP_STATE_DIR;
  process.env.OPENKLIP_STATE_DIR = root;
  try {
    const referenceImagePaths = ["a.jpg", "b.jpg", "c.jpg"].map(
      (name, index) => {
        const path = join(root, name);
        writeFileSync(path, `reference-${index}`);
        return path;
      }
    );
    const negativeReferenceImagePaths = [join(root, "not-member.jpg")];
    writeFileSync(negativeReferenceImagePaths[0], "hard-negative");
    const runWorker = async (
      images: Array<{ id: string; path: string }>
    ): Promise<FaceWorkerResult> => ({
      version: 1,
      detectorModel: "detector-test",
      recognizerModel: "recognizer-test",
      images: images.map((image, index) => ({
        id: image.id,
        path: image.path,
        atSec: null,
        width: 400,
        height: 400,
        faces: [
          {
            box: { x: 100, y: 100, width: 150, height: 150 },
            confidence: 0.99,
            landmarks: [],
            sharpness: 100,
            vectorB64: Buffer.from(
              Float32Array.from([1, index, 0, 0]).buffer
            ).toString("base64"),
          },
        ],
      })),
    });
    const profile = await createMemberProfile(
      {
        id: "member:nmixx:haewon",
        displayName: "Haewon",
        groupId: "nmixx",
        referenceImagePaths,
        negativeReferenceImagePaths,
      },
      { runWorker }
    );
    assert.equal(profile.references.length, 3);
    assert.equal(profile.negativeReferences.length, 1);
    assert.equal(profile.calibration.status, "hard-negatives-present");
    assert.equal(profile.calibration.hardNegativeCount, 1);
    assert.equal(profile.calibration.thresholdSource, "default-unmeasured");
    assert.match(profile.fingerprint, /^[a-f0-9]{64}$/);
    assert.equal(loadMemberProfile(profile.id).displayName, "Haewon");
    assert.equal(listMemberProfiles().length, 1);
    assert.equal(
      existsSync(join(memberProfileDir(profile.id), "profile.json")),
      true
    );
  } finally {
    if (previousState === undefined) {
      delete process.env.OPENKLIP_STATE_DIR;
    } else {
      process.env.OPENKLIP_STATE_DIR = previousState;
    }
    await rm(root, { force: true, recursive: true });
  }
});

test("createMemberProfile rejects fewer than three references before inference", async () => {
  await assert.rejects(
    createMemberProfile({
      id: "member:test:one",
      displayName: "One",
      referenceImagePaths: ["a.jpg"],
    }),
    /requires 3-40 reference images/
  );
});

test("createMemberProfileFromSources rejects non-HTTPS remote references", async () => {
  await assert.rejects(
    createMemberProfileFromSources({
      id: "member:test:remote",
      displayName: "Remote",
      referenceImageUrls: [
        "http://example.com/a.jpg",
        "http://example.com/b.jpg",
        "http://example.com/c.jpg",
      ],
    }),
    /must use HTTPS/
  );
});

test("readResponseBodyLimited streams and rejects bytes beyond the cap", async () => {
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1, 2, 3]));
        controller.enqueue(Uint8Array.from([4, 5, 6]));
        controller.close();
      },
    })
  );
  await assert.rejects(readResponseBodyLimited(response, 5), /20 MB limit/);
});

test("readResponseBodyLimited returns an incrementally-read payload", async () => {
  const response = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1, 2]));
        controller.enqueue(Uint8Array.from([3]));
        controller.close();
      },
    })
  );
  assert.deepEqual(
    [...(await readResponseBodyLimited(response, 3))],
    [1, 2, 3]
  );
});
