import { existsSync, readdirSync } from "node:fs";
import { mkdir, rm } from "node:fs/promises";
import { join } from "node:path";
import { FFMPEG, ProcessCancelledError, run } from "./ffmpeg.ts";
import type { IndexedFaceHit } from "./media-index-core.ts";

const FACE_REFINE_FPS = 10;
const FACE_FRAME_STEP_SEC = 0.25;

export interface TimedFrame {
  atSec: number;
  path: string;
}

export interface TimeWindow {
  endSec: number;
  startSec: number;
}

function sortedJpegs(directory: string): string[] {
  if (!existsSync(directory)) {
    return [];
  }
  return readdirSync(directory)
    .filter((name) => /\.jpe?g$/i.test(name))
    .sort()
    .map((name) => join(directory, name));
}

export async function extractFrames(
  source: string,
  directory: string,
  fps: number,
  signal?: AbortSignal
): Promise<string[]> {
  await rm(directory, { force: true, recursive: true });
  await mkdir(directory, { recursive: true });
  await run(
    FFMPEG,
    [
      "-y",
      "-i",
      source,
      "-vf",
      `fps=${fps},scale=640:-2`,
      "-q:v",
      "6",
      join(directory, "%07d.jpg"),
    ],
    `ffmpeg(media-index-${fps}fps)`,
    signal
  );
  return sortedJpegs(directory);
}

async function runFfmpegCapture(
  args: string[],
  label: string,
  signal?: AbortSignal
): Promise<string> {
  if (signal?.aborted) {
    throw new ProcessCancelledError(label);
  }
  const processHandle = Bun.spawn([FFMPEG, ...args], {
    stdout: "pipe",
    stderr: "pipe",
  });
  let onAbort: (() => void) | undefined;
  if (signal) {
    onAbort = () => processHandle.kill();
    signal.addEventListener("abort", onAbort, { once: true });
  }
  try {
    const [stderr, exitCode] = await Promise.all([
      new Response(processHandle.stderr).text(),
      processHandle.exited,
    ]);
    if (exitCode !== 0) {
      if (signal?.aborted) {
        throw new ProcessCancelledError(label);
      }
      throw new Error(`${label} failed with exit code ${exitCode}`);
    }
    return stderr;
  } finally {
    if (signal && onAbort) {
      signal.removeEventListener("abort", onAbort);
    }
  }
}

export async function extractSceneCutFrames(
  source: string,
  directory: string,
  signal?: AbortSignal
): Promise<TimedFrame[]> {
  const stderr = await runFfmpegCapture(
    [
      "-y",
      "-i",
      source,
      "-vf",
      "select='gt(scene,0.28)',showinfo,scale=640:-2",
      "-fps_mode",
      "vfr",
      "-q:v",
      "6",
      join(directory, "cut-%07d.jpg"),
    ],
    "ffmpeg(media-index-scene-cuts)",
    signal
  );
  const times = Array.from(stderr.matchAll(/pts_time:([0-9]+(?:\.[0-9]+)?)/g))
    .map((match) => Number(match[1]))
    .filter(Number.isFinite);
  const paths = readdirSync(directory)
    .filter((name) => /^cut-\d+\.jpe?g$/i.test(name))
    .sort()
    .map((name) => join(directory, name));
  return paths.slice(0, times.length).map((path, index) => ({
    path,
    atSec: times[index],
  }));
}

function mergeTimeWindows(windows: TimeWindow[]): TimeWindow[] {
  const sorted = windows
    .filter((window) => window.endSec > window.startSec)
    .sort((left, right) => left.startSec - right.startSec);
  const merged: TimeWindow[] = [];
  for (const window of sorted) {
    const previous = merged.at(-1);
    if (previous && window.startSec <= previous.endSec + 0.1) {
      previous.endSec = Math.max(previous.endSec, window.endSec);
    } else {
      merged.push({ ...window });
    }
  }
  return merged;
}

export function refinementWindows(
  hits: IndexedFaceHit[],
  cutTimes: number[],
  durationSec: number
): TimeWindow[] {
  const windows: TimeWindow[] = cutTimes.map((cut) => ({
    startSec: Math.max(0, cut - 0.5),
    endSec: Math.min(durationSec, cut + 0.5),
  }));
  for (const hit of hits) {
    if (hit.status !== "matched") {
      windows.push({
        startSec: Math.max(0, hit.atSec - 0.75),
        endSec: Math.min(durationSec, hit.atSec + 0.75),
      });
    }
  }
  const matched = hits.filter((hit) => hit.status === "matched");
  if (matched.length > 0) {
    const first = matched[0].atSec;
    const last = (matched.at(-1) as IndexedFaceHit).atSec;
    windows.push(
      {
        startSec: Math.max(0, first - 0.5),
        endSec: Math.min(durationSec, first + 0.5),
      },
      {
        startSec: Math.max(0, last - 0.5),
        endSec: Math.min(durationSec, last + 0.5),
      }
    );
    for (let index = 1; index < matched.length; index += 1) {
      const previous = matched[index - 1];
      const current = matched[index];
      if (current.atSec - previous.atSec <= FACE_FRAME_STEP_SEC * 1.5) {
        continue;
      }
      windows.push(
        {
          startSec: Math.max(0, previous.atSec - 0.5),
          endSec: Math.min(durationSec, previous.atSec + 0.5),
        },
        {
          startSec: Math.max(0, current.atSec - 0.5),
          endSec: Math.min(durationSec, current.atSec + 0.5),
        }
      );
    }
  }
  return mergeTimeWindows(windows);
}

export async function extractRefinementFrames(
  source: string,
  directory: string,
  windows: TimeWindow[],
  signal?: AbortSignal
): Promise<TimedFrame[]> {
  const frames: TimedFrame[] = [];
  for (const [windowIndex, window] of windows.entries()) {
    const windowDirectory = join(
      directory,
      `refine-${String(windowIndex + 1).padStart(4, "0")}`
    );
    await mkdir(windowDirectory, { recursive: true });
    await run(
      FFMPEG,
      [
        "-y",
        "-ss",
        String(window.startSec),
        "-t",
        String(window.endSec - window.startSec),
        "-i",
        source,
        "-vf",
        `fps=${FACE_REFINE_FPS},scale=640:-2`,
        "-q:v",
        "6",
        join(windowDirectory, "%07d.jpg"),
      ],
      "ffmpeg(media-index-face-refine)",
      signal
    );
    for (const [index, path] of sortedJpegs(windowDirectory).entries()) {
      frames.push({
        path,
        atSec: window.startSec + index / FACE_REFINE_FPS,
      });
    }
  }
  return frames;
}

export function replaceHitsInWindows(
  coarseHits: IndexedFaceHit[],
  refinedHits: IndexedFaceHit[],
  windows: TimeWindow[]
): IndexedFaceHit[] {
  const kept = coarseHits.filter(
    (hit) =>
      !windows.some(
        (window) => hit.atSec >= window.startSec && hit.atSec <= window.endSec
      )
  );
  return [...kept, ...refinedHits].sort(
    (left, right) => left.atSec - right.atSec
  );
}
