// URL ingester runner: download remote video with yt-dlp (manifest in
// ingesters/url/), then hand the file to the normal ingest path.
import { existsSync, readdirSync } from "node:fs";
import { delimiter, extname, join } from "node:path";
import { run } from "./ffmpeg.ts";
import { loadIngesters, resolveIngesterArgv } from "./ingesters.ts";
import {
  isSupportedVideoFilename,
  unsupportedVideoMessage,
} from "./video-formats.ts";

export class UrlIngesterUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "UrlIngesterUnavailableError";
  }
}

function commandOnPath(command: string): boolean {
  const pathEntries = (process.env.PATH ?? "").split(delimiter).filter(Boolean);
  const extensions =
    process.platform === "win32" && !extname(command)
      ? (process.env.PATHEXT ?? ".COM;.EXE;.BAT;.CMD")
          .split(";")
          .filter(Boolean)
      : [""];
  for (const directory of pathEntries) {
    for (const extension of extensions) {
      if (existsSync(join(directory, `${command}${extension}`))) {
        return true;
      }
    }
  }
  return false;
}

function pickDownloadedVideo(tmpDir: string): string {
  const candidates = readdirSync(tmpDir)
    .map((name) => ({ name, abs: join(tmpDir, name) }))
    .filter(
      (entry) =>
        existsSync(entry.abs) &&
        !entry.name.startsWith(".") &&
        isSupportedVideoFilename(entry.name)
    );
  if (candidates.length === 0) {
    throw new Error(
      "URL download finished but no supported video file was found in the temp dir."
    );
  }
  candidates.sort((a, b) => b.name.length - a.name.length);
  return candidates[0].abs;
}

/** Download a remote video to tmpDir; returns the absolute path to the file. */
export async function downloadVideoFromUrl(
  url: string,
  tmpDir: string,
  signal?: AbortSignal
): Promise<string> {
  const trimmed = url.trim();
  if (!trimmed) {
    throw new Error("URL is required.");
  }
  const ingesters = await loadIngesters();
  const manifest = ingesters.find((entry) => entry.id === "url");
  if (!manifest) {
    throw new UrlIngesterUnavailableError(
      "URL ingester manifest missing (ingesters/url/ingester.json)."
    );
  }
  if (!commandOnPath(manifest.command)) {
    throw new UrlIngesterUnavailableError(
      `${manifest.command} is not on PATH. Install yt-dlp to import from URLs.`
    );
  }
  const outputTemplate = join(tmpDir, "download.%(ext)s");
  const argv = resolveIngesterArgv(manifest, { url: trimmed }, outputTemplate);
  const [command, ...args] = argv;
  await run(command, args, "yt-dlp", signal);
  const downloaded = pickDownloadedVideo(tmpDir);
  if (!isSupportedVideoFilename(downloaded)) {
    throw new Error(unsupportedVideoMessage(downloaded));
  }
  return downloaded;
}
