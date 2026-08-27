import assert from "node:assert/strict";
import { test } from "node:test";
import {
  refinementWindows,
  replaceHitsInWindows,
} from "../src/media-index-frames.ts";
import type { IndexedFaceHit } from "../src/media-index-core.ts";

function face(atSec: number, status: IndexedFaceHit["status"]): IndexedFaceHit {
  return {
    atSec,
    box: { x: 0, y: 0, width: 100, height: 100 },
    confidence: 0.99,
    margin: status === "matched" ? 0.2 : 0.01,
    memberProfileId: "member:test",
    score: 0.8,
    status,
    vector: Float32Array.from([1, 0]),
  };
}

test("refinementWindows covers shot cuts, ambiguity, and internal gaps", () => {
  const windows = refinementWindows(
    [
      face(1, "matched"),
      face(1.25, "matched"),
      face(3, "matched"),
      face(5, "ambiguous"),
    ],
    [7],
    10
  );

  assert.ok(
    windows.some((window) => window.startSec <= 2.5 && window.endSec >= 3.5)
  );
  assert.ok(
    windows.some((window) => window.startSec <= 4.25 && window.endSec >= 5.75)
  );
  assert.ok(
    windows.some((window) => window.startSec <= 6.5 && window.endSec >= 7.5)
  );
});

test("replaceHitsInWindows swaps coarse samples for refined evidence", () => {
  const result = replaceHitsInWindows(
    [face(0, "matched"), face(1, "ambiguous"), face(2, "matched")],
    [face(0.9, "matched"), face(1.1, "matched")],
    [{ startSec: 0.5, endSec: 1.5 }]
  );

  assert.deepEqual(
    result.map((hit) => hit.atSec),
    [0, 0.9, 1.1, 2]
  );
});
