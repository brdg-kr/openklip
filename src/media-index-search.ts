import { bytesToVector, cosineSimilarity } from "./media-index-core.ts";
import type { StoredSceneEmbedding } from "./media-index-storage.ts";

const MAX_SCENE_RESULTS = 100;
const SCENE_PEAK_MARGIN = 0.04;
const SCENE_FUSION_PEAK_RATIO = 0.98;

export interface SceneQueryVector {
  query: string;
  vector: Float32Array;
}

export interface MediaSearchResult {
  atSec: number;
  confidence: number;
  fromSec: number;
  fusedScore: number;
  memberProfileId?: string;
  queryScores: Array<{ query: string; rank: number; score: number }>;
  scopes: string[];
  thumbnail: string;
  toSec: number;
}

function decodeVector(encoded: string): Float32Array {
  return bytesToVector(Buffer.from(encoded, "base64"));
}

function mergeQueryScores(
  left: MediaSearchResult["queryScores"],
  right: MediaSearchResult["queryScores"]
): MediaSearchResult["queryScores"] {
  const byQuery = new Map(left.map((item) => [item.query, item]));
  for (const item of right) {
    const previous = byQuery.get(item.query);
    if (!previous || item.score > previous.score) {
      byQuery.set(item.query, item);
    }
  }
  return [...byQuery.values()];
}

function mergeResult(
  target: MediaSearchResult,
  incoming: MediaSearchResult
): void {
  target.fromSec = Math.min(target.fromSec, incoming.fromSec);
  target.toSec = Math.max(target.toSec, incoming.toSec);
  target.fusedScore = Math.max(target.fusedScore, incoming.fusedScore);
  target.confidence = Math.max(target.confidence, incoming.confidence);
  target.scopes = Array.from(new Set([...target.scopes, ...incoming.scopes]));
  target.queryScores = mergeQueryScores(
    target.queryScores,
    incoming.queryScores
  );
}

export function rankMediaScenes(
  allRows: StoredSceneEmbedding[],
  queryVectors: SceneQueryVector[],
  options: { limit?: number; memberProfileId?: string } = {}
): MediaSearchResult[] {
  if (queryVectors.length === 0 || queryVectors.length > 3) {
    throw new Error("scene search requires one to three query vectors");
  }
  const rows = allRows.filter((row) => {
    if (options.memberProfileId) {
      return row.memberProfileId === options.memberProfileId;
    }
    return row.scope === "full";
  });
  if (rows.length === 0) {
    return [];
  }
  const queryRankings = queryVectors.map(({ query, vector }) => {
    const ranked = rows
      .map((row) => ({
        id: row.id,
        query,
        score: cosineSimilarity(decodeVector(row.vectorB64), vector),
      }))
      .sort((left, right) => right.score - left.score);
    return new Map(
      ranked.map((item, index) => [
        item.id,
        { query: item.query, rank: index + 1, score: item.score },
      ])
    );
  });
  const fused = rows.map((row): MediaSearchResult => {
    const queryScores = queryRankings.map((ranking) => {
      const item = ranking.get(row.id);
      if (!item) {
        throw new Error(`scene ranking is missing row ${row.id}`);
      }
      return item;
    });
    return {
      atSec: row.atSec,
      fromSec: row.fromSec,
      toSec: row.toSec,
      scopes: [row.scope],
      thumbnail: row.thumbnail,
      confidence: Math.max(...queryScores.map((item) => item.score)),
      fusedScore: queryScores.reduce(
        (sum, item) => sum + 1 / (60 + item.rank),
        0
      ),
      queryScores,
      ...(row.memberProfileId ? { memberProfileId: row.memberProfileId } : {}),
    };
  });
  const peak = Math.max(...fused.map((row) => row.confidence));
  const fusedPeak = Math.max(...fused.map((row) => row.fusedScore));
  const sameTime = new Map<string, MediaSearchResult>();
  for (const row of fused.filter(
    (item) =>
      item.confidence >= peak - SCENE_PEAK_MARGIN ||
      item.fusedScore >= fusedPeak * SCENE_FUSION_PEAK_RATIO
  )) {
    const key = row.atSec.toFixed(3);
    const previous = sameTime.get(key);
    if (previous) {
      mergeResult(previous, row);
    } else {
      sameTime.set(key, { ...row });
    }
  }
  const temporal = [...sameTime.values()].sort(
    (left, right) => left.atSec - right.atSec
  );
  const clustered: MediaSearchResult[] = [];
  for (const row of temporal) {
    const previous = clustered.at(-1);
    if (previous && row.fromSec - previous.toSec <= 0.1) {
      mergeResult(previous, row);
      continue;
    }
    clustered.push({ ...row });
  }
  const limit = Math.min(MAX_SCENE_RESULTS, Math.max(1, options.limit ?? 24));
  return clustered
    .sort(
      (left, right) =>
        right.fusedScore - left.fusedScore || left.fromSec - right.fromSec
    )
    .slice(0, limit);
}
