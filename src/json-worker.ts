import { randomUUID } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { ProcessCancelledError } from "./ffmpeg.ts";

export interface RunJsonWorkerInput<TJob> {
  args: (jobPath: string, outputPath: string) => string[];
  job: TJob;
  label: string;
  signal?: AbortSignal;
  temporaryDirectory: string;
}

export async function runJsonWorker<TJob, TResult>({
  args,
  job,
  label,
  signal,
  temporaryDirectory,
}: RunJsonWorkerInput<TJob>): Promise<TResult> {
  if (signal?.aborted) {
    throw new ProcessCancelledError(label);
  }
  await mkdir(temporaryDirectory, { recursive: true });
  const nonce = randomUUID();
  const jobPath = join(temporaryDirectory, `${label}-job-${nonce}.json`);
  const outputPath = join(temporaryDirectory, `${label}-result-${nonce}.json`);
  await writeFile(jobPath, JSON.stringify(job));
  const processHandle = Bun.spawn(["node", ...args(jobPath, outputPath)], {
    stdout: "pipe",
    stderr: "inherit",
  });
  let onAbort: (() => void) | undefined;
  if (signal) {
    onAbort = () => processHandle.kill();
    signal.addEventListener("abort", onAbort, { once: true });
  }
  try {
    const exitCode = await processHandle.exited;
    if (exitCode !== 0) {
      if (signal?.aborted) {
        throw new ProcessCancelledError(label);
      }
      throw new Error(`${label} worker failed with exit code ${exitCode}`);
    }
    return JSON.parse(await readFile(outputPath, "utf8")) as TResult;
  } finally {
    if (signal && onAbort) {
      signal.removeEventListener("abort", onAbort);
    }
    await Promise.all([
      rm(jobPath, { force: true }),
      rm(outputPath, { force: true }),
    ]);
  }
}
