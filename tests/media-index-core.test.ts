import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildMemberAppearanceTracks,
  cosineSimilarity,
  type IndexedFaceHit,
  matchFaceVector,
} from "../src/media-index-core.ts";

function hit(
  atSec: number,
  score: number,
  status: IndexedFaceHit["status"] = "matched"
): IndexedFaceHit {
  return {
    atSec,
    box: { x: 10, y: 10, width: 100, height: 100 },
    confidence: 0.95,
    margin: 0.2,
    memberProfileId: "member:nmixx:haewon",
    score,
    status,
    vector: Float32Array.from([1, 0]),
  };
}

test("cosineSimilarity distinguishes aligned and opposite face vectors", () => {
  assert.equal(
    cosineSimilarity(Float32Array.from([1, 0]), Float32Array.from([1, 0])),
    1
  );
  assert.equal(
    cosineSimilarity(Float32Array.from([1, 0]), Float32Array.from([-1, 0])),
    -1
  );
});

test("matchFaceVector requires absolute score and hard-negative margin", () => {
  const target = [
    {
      memberProfileId: "member:nmixx:haewon",
      vector: Float32Array.from([1, 0]),
    },
  ];
  const negatives = [
    {
      memberProfileId: "member:nmixx:sullyoon",
      vector: Float32Array.from([0.99, 0.1]),
    },
  ];
  const ambiguous = matchFaceVector(
    Float32Array.from([1, 0]),
    target,
    negatives
  );
  assert.equal(ambiguous.status, "ambiguous");
  assert.equal(ambiguous.memberProfileId, "member:nmixx:haewon");

  const matched = matchFaceVector(Float32Array.from([1, 0]), target, [
    {
      memberProfileId: "member:nmixx:sullyoon",
      vector: Float32Array.from([0, 1]),
    },
  ]);
  assert.equal(matched.status, "matched");
});

test("matchFaceVector never confirms identity without hard negatives", () => {
  const result = matchFaceVector(
    Float32Array.from([1, 0]),
    [
      {
        memberProfileId: "member:nmixx:haewon",
        vector: Float32Array.from([1, 0]),
      },
    ],
    []
  );
  assert.equal(result.status, "ambiguous");
  assert.equal(result.margin, 0);
});

test("a high-score ambiguous hit never becomes a matched track", () => {
  const tracks = buildMemberAppearanceTracks([hit(0, 0.99, "ambiguous")], {
    frameStepSec: 0.1,
  });
  assert.equal(tracks[0].status, "ambiguous");
});

test("multiple detections in one frame count as one track observation", () => {
  const tracks = buildMemberAppearanceTracks([hit(0, 0.9), hit(0, 0.8)], {
    frameStepSec: 0.25,
  });
  assert.equal(tracks[0].hitCount, 1);
});

test("buildMemberAppearanceTracks aggregates adjacent evidence and preserves ambiguity", () => {
  const tracks = buildMemberAppearanceTracks(
    [hit(0, 0.45), hit(0.25, 0.47), hit(2, 0.38, "ambiguous")],
    { frameStepSec: 0.25 }
  );
  assert.equal(tracks.length, 2);
  assert.equal(tracks[0].status, "matched");
  assert.equal(tracks[0].hitCount, 2);
  assert.equal(tracks[0].startSec, 0);
  assert.equal(tracks[0].endSec, 0.375);
  assert.equal(tracks[1].status, "ambiguous");
});
