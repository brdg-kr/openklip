import assert from "node:assert/strict";
import { test } from "node:test";
import { vectorToBytes } from "../src/media-index-core.ts";
import { rankMediaScenes } from "../src/media-index-search.ts";
import type { StoredSceneEmbedding } from "../src/media-index-storage.ts";

function vectorB64(values: number[]): string {
  return Buffer.from(vectorToBytes(Float32Array.from(values))).toString(
    "base64"
  );
}

function scene(
  id: string,
  atSec: number,
  scope: string,
  vector: number[],
  memberProfileId?: string
): StoredSceneEmbedding {
  return {
    id,
    atSec,
    fromSec: atSec,
    toSec: atSec + 1,
    scope,
    thumbnail: `/frames/${id}.jpg`,
    vectorB64: vectorB64(vector),
    ...(memberProfileId ? { memberProfileId } : {}),
  };
}

test("rankMediaScenes fuses Korean and English queries across target scopes", () => {
  const memberProfileId = "member:nmixx:haewon";
  const rows = [
    scene("full-1", 2, "full", [1, 0], memberProfileId),
    scene("context-1", 2, "target-context", [0.99, 0.01], memberProfileId),
    scene("medium-1", 2, "target-medium", [0.98, 0.02], memberProfileId),
    scene("other", 8, "full", [0, 1], memberProfileId),
    scene("different-member", 2, "full", [1, 0], "member:nmixx:lily"),
  ];

  const results = rankMediaScenes(
    rows,
    [
      { query: "놀라서 웃는 장면", vector: Float32Array.from([1, 0]) },
      {
        query: "a surprised reaction followed by laughter",
        vector: Float32Array.from([0.95, 0.05]),
      },
    ],
    { memberProfileId, limit: 5 }
  );

  assert.equal(results.length, 1);
  assert.deepEqual(results[0].scopes.sort(), [
    "full",
    "target-context",
    "target-medium",
  ]);
  assert.deepEqual(
    results[0].queryScores.map((item) => item.query),
    ["놀라서 웃는 장면", "a surprised reaction followed by laughter"]
  );
});

test("rankMediaScenes without a member profile searches whole frames only", () => {
  const results = rankMediaScenes(
    [
      scene("full", 1, "full", [1, 0]),
      scene("crop", 1, "target-medium", [1, 0], "member:test"),
    ],
    [{ query: "무대", vector: Float32Array.from([1, 0]) }]
  );

  assert.equal(results.length, 1);
  assert.deepEqual(results[0].scopes, ["full"]);
});

test("rankMediaScenes rejects more than three query embeddings", () => {
  assert.throws(
    () =>
      rankMediaScenes(
        [scene("full", 1, "full", [1, 0])],
        ["a", "b", "c", "d"].map((query) => ({
          query,
          vector: Float32Array.from([1, 0]),
        }))
      ),
    /one to three query vectors/
  );
});
