export type FaceMatchStatus = "ambiguous" | "matched" | "unknown";

export interface FaceBox {
  height: number;
  width: number;
  x: number;
  y: number;
}

export interface IndexedFaceHit {
  atSec: number;
  box: FaceBox;
  confidence: number;
  margin: number;
  memberProfileId: string | null;
  score: number;
  status: FaceMatchStatus;
  vector: Float32Array;
}

export interface MemberAppearanceTrack {
  confidence: number;
  endSec: number;
  hitCount: number;
  id: string;
  margin: number;
  memberProfileId: string;
  startSec: number;
  status: "ambiguous" | "matched";
}

export interface ReferenceVector {
  memberProfileId: string;
  vector: Float32Array;
}

export interface FaceMatchOptions {
  ambiguousScore: number;
  matchMargin: number;
  matchScore: number;
}

export const DEFAULT_FACE_MATCH_OPTIONS: FaceMatchOptions = {
  matchScore: 0.4,
  ambiguousScore: 0.3,
  matchMargin: 0.06,
};

export function cosineSimilarity(
  left: Float32Array,
  right: Float32Array
): number {
  if (left.length !== right.length || left.length === 0) {
    return Number.NEGATIVE_INFINITY;
  }
  let dot = 0;
  let leftSquares = 0;
  let rightSquares = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index];
    const rightValue = right[index];
    dot += leftValue * rightValue;
    leftSquares += leftValue * leftValue;
    rightSquares += rightValue * rightValue;
  }
  const denominator = Math.sqrt(leftSquares) * Math.sqrt(rightSquares);
  return denominator > 0 ? dot / denominator : Number.NEGATIVE_INFINITY;
}

function bestScore(
  vector: Float32Array,
  references: ReferenceVector[]
): { memberProfileId: string | null; score: number } {
  let memberProfileId: string | null = null;
  let score = Number.NEGATIVE_INFINITY;
  for (const reference of references) {
    const candidate = cosineSimilarity(vector, reference.vector);
    if (candidate > score) {
      score = candidate;
      memberProfileId = reference.memberProfileId;
    }
  }
  return { memberProfileId, score };
}

export function matchFaceVector(
  vector: Float32Array,
  targetReferences: ReferenceVector[],
  negativeReferences: ReferenceVector[],
  options: FaceMatchOptions = DEFAULT_FACE_MATCH_OPTIONS
): Pick<IndexedFaceHit, "margin" | "memberProfileId" | "score" | "status"> {
  const target = bestScore(vector, targetReferences);
  if (!(target.memberProfileId && Number.isFinite(target.score))) {
    return {
      margin: 0,
      memberProfileId: null,
      score: 0,
      status: "unknown",
    };
  }
  const negative = bestScore(vector, negativeReferences);
  if (!Number.isFinite(negative.score)) {
    return {
      margin: 0,
      memberProfileId: target.memberProfileId,
      score: target.score,
      status: target.score >= options.ambiguousScore ? "ambiguous" : "unknown",
    };
  }
  const negativeScore = negative.score;
  const margin = target.score - negativeScore;
  if (target.score >= options.matchScore && margin >= options.matchMargin) {
    return {
      margin,
      memberProfileId: target.memberProfileId,
      score: target.score,
      status: "matched",
    };
  }
  if (target.score >= options.ambiguousScore) {
    return {
      margin,
      memberProfileId: target.memberProfileId,
      score: target.score,
      status: "ambiguous",
    };
  }
  return {
    margin,
    memberProfileId: null,
    score: target.score,
    status: "unknown",
  };
}

export interface BuildTracksOptions {
  frameStepSec: number;
  maxGapSec?: number;
  singleHitHighScore?: number;
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

export function buildMemberAppearanceTracks(
  hits: IndexedFaceHit[],
  options: BuildTracksOptions
): MemberAppearanceTrack[] {
  const maxGapSec = options.maxGapSec ?? options.frameStepSec * 2.5;
  const singleHitHighScore = options.singleHitHighScore ?? 0.55;
  const bestPerFrame = new Map<
    string,
    IndexedFaceHit & { memberProfileId: string }
  >();
  for (const hit of hits.filter(
    (candidate): candidate is IndexedFaceHit & { memberProfileId: string } =>
      candidate.status !== "unknown" && candidate.memberProfileId !== null
  )) {
    const key = `${hit.memberProfileId}:${hit.atSec.toFixed(6)}`;
    const previous = bestPerFrame.get(key);
    if (
      !previous ||
      (hit.status === "matched" && previous.status !== "matched") ||
      (hit.status === previous.status && hit.score > previous.score)
    ) {
      bestPerFrame.set(key, hit);
    }
  }
  const candidates = [...bestPerFrame.values()].sort(
    (left, right) =>
      left.memberProfileId.localeCompare(right.memberProfileId) ||
      left.atSec - right.atSec ||
      right.score - left.score
  );
  const groups: Array<Array<IndexedFaceHit & { memberProfileId: string }>> = [];
  for (const hit of candidates) {
    const current = groups.at(-1);
    const previous = current?.at(-1);
    if (
      previous &&
      current &&
      previous.memberProfileId === hit.memberProfileId &&
      hit.atSec - previous.atSec <= maxGapSec
    ) {
      current.push(hit);
    } else {
      groups.push([hit]);
    }
  }
  return groups.map((group, index) => {
    const scores = group.map((hit) => hit.score);
    const margins = group.map((hit) => hit.margin);
    const matchedCount = group.filter((hit) => hit.status === "matched").length;
    const status =
      matchedCount >= 2 ||
      (group.length === 1 &&
        group[0].status === "matched" &&
        group[0].score >= singleHitHighScore)
        ? "matched"
        : "ambiguous";
    const first = group[0];
    const last = group.at(-1) as (typeof group)[number];
    const halfStep = options.frameStepSec / 2;
    return {
      id: `face-track-${index + 1}`,
      memberProfileId: first.memberProfileId,
      startSec: Math.max(0, first.atSec - halfStep),
      endSec: last.atSec + halfStep,
      hitCount: group.length,
      confidence: median(scores),
      margin: median(margins),
      status,
    };
  });
}

export function vectorToBytes(vector: Float32Array): Uint8Array {
  return new Uint8Array(
    vector.buffer.slice(
      vector.byteOffset,
      vector.byteOffset + vector.byteLength
    )
  );
}

export function bytesToVector(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("face vector byte length is not divisible by four");
  }
  const copy = Uint8Array.from(bytes);
  return new Float32Array(copy.buffer);
}
