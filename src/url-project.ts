import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import type { IngestOpts } from "./ingest.ts";
import { ingest } from "./ingest.ts";
import {
  getIngestJob,
  type IngestJob,
  releaseIngestSlug,
  reserveIngestSlug,
  startIngestJob,
} from "./ingest-jobs.ts";
import {
  IngestPartialError,
  IngestPersistError,
} from "./ingest-persist-error.ts";
import { buildMediaIndex } from "./media-index.ts";
import { loadMemberProfile } from "./member-profiles.ts";
import { assertValidSlug, projectPaths, slugify } from "./paths.ts";
import { persistProjectSource } from "./project-source.ts";
import { downloadVideoFromUrl } from "./url-ingest.ts";

const HTTP_PROTOCOLS = new Set(["http:", "https:"]);
const YOUTUBE_HOSTS = new Set([
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "music.youtube.com",
  "youtube-nocookie.com",
  "www.youtube-nocookie.com",
  "youtu.be",
]);
const YOUTUBE_ID = /^[A-Za-z0-9_-]{6,64}$/;

export interface UrlProjectCreateInput {
  force?: boolean;
  memberProfileId?: string;
  projectSlug?: string;
  tempRoot?: string;
  url: string;
}

export interface UrlProjectCreateResult {
  filename: string;
  projectDir: string;
  slug: string;
  sourcePath: string;
}

export interface UrlProjectJobResult {
  jobId: string;
  slug: string;
  status: IngestJob["status"];
}

export type ProjectIngest = (
  videoArg: string,
  opts?: Pick<IngestOpts, "force" | "onProgress" | "signal" | "slug">
) => Promise<string>;

export interface DownloadedUrlProjectInput {
  downloadedPath: string;
  force?: boolean;
  ingest: ProjectIngest;
  onProgress?: IngestOpts["onProgress"];
  signal?: AbortSignal;
  slug: string;
}

interface RunUrlProjectInput extends UrlProjectCreateInput {
  ingest?: ProjectIngest;
  onProgress?: IngestOpts["onProgress"];
  signal?: AbortSignal;
  slug: string;
}

export interface StartUrlProjectDeps {
  run?: (input: RunUrlProjectInput) => Promise<UrlProjectCreateResult>;
}

function parseHttpUrl(raw: string): URL {
  const trimmed = raw.trim();
  if (!trimmed) {
    throw new Error("URL is required.");
  }
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error("URL must be a valid HTTP or HTTPS URL.");
  }
  if (!HTTP_PROTOCOLS.has(parsed.protocol)) {
    throw new Error("URL must use HTTP or HTTPS.");
  }
  if (parsed.username || parsed.password) {
    throw new Error("URL must not contain embedded credentials.");
  }
  return parsed;
}

function youtubeVideoId(parsed: URL): string | null {
  const host = parsed.hostname.toLowerCase();
  if (!YOUTUBE_HOSTS.has(host)) {
    return null;
  }
  let candidate = "";
  if (host === "youtu.be") {
    candidate = parsed.pathname.split("/").filter(Boolean)[0] ?? "";
  } else if (parsed.pathname === "/watch") {
    candidate = parsed.searchParams.get("v") ?? "";
  } else {
    const parts = parsed.pathname.split("/").filter(Boolean);
    if (["embed", "live", "shorts"].includes(parts[0] ?? "")) {
      candidate = parts[1] ?? "";
    }
  }
  return YOUTUBE_ID.test(candidate) ? candidate : null;
}

/** Stable, collision-resistant slug for browser and Agent URL intake. */
export function projectSlugFromUrl(
  rawUrl: string,
  requestedSlug?: string
): string {
  const parsed = parseHttpUrl(rawUrl);
  if (requestedSlug?.trim()) {
    return assertValidSlug(requestedSlug.trim());
  }
  const videoId = youtubeVideoId(parsed);
  if (videoId) {
    return assertValidSlug(`youtube-${videoId}`);
  }
  const digest = createHash("sha256")
    .update(parsed.toString())
    .digest("hex")
    .slice(0, 10);
  const pathName = parsed.pathname.split("/").filter(Boolean).at(-1) ?? "video";
  const descriptive = slugify(`${parsed.hostname}-${pathName}`).slice(0, 49);
  return assertValidSlug(`url-${descriptive}-${digest}`);
}

export function sanitizeUrlProjectFilename(name: string): string {
  return basename(name).replace(/[^\w.-]+/g, "_") || "video.mp4";
}

export function safeUrlDownloadErrorMessage(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("is not on PATH")) {
    return message;
  }
  if (/HTTP(?: Error)? 403/i.test(message)) {
    return "Could not download this video: the provider returned HTTP 403.";
  }
  if (/video is unavailable|error code:\s*152/i.test(message)) {
    return "Could not download this video: YouTube did not make it available to the configured downloader.";
  }
  return "Could not download this video URL. Check the URL and try again.";
}

/** Complete ingest after a URL downloader has produced a local media file. */
export async function ingestDownloadedUrlProject({
  downloadedPath,
  force,
  ingest: ingestProject,
  onProgress,
  signal,
  slug,
}: DownloadedUrlProjectInput): Promise<UrlProjectCreateResult> {
  const filename = sanitizeUrlProjectFilename(basename(downloadedPath));
  const createdSlug = await ingestProject(downloadedPath, {
    force,
    onProgress,
    signal,
    slug,
  });
  if (createdSlug !== slug) {
    throw new Error(
      `URL ingest created unexpected project "${createdSlug}" instead of "${slug}".`
    );
  }
  let sourcePath: string;
  try {
    sourcePath = await persistProjectSource(
      createdSlug,
      filename,
      downloadedPath
    );
  } catch (error) {
    throw new IngestPersistError(createdSlug, error);
  }
  return {
    filename,
    projectDir: projectPaths(createdSlug).dir,
    slug: createdSlug,
    sourcePath,
  };
}

async function runUrlProject({
  force,
  ingest: ingestProject = ingest,
  memberProfileId,
  onProgress,
  signal,
  slug,
  tempRoot,
  url,
}: RunUrlProjectInput): Promise<UrlProjectCreateResult> {
  const temporaryDirectory = await mkdtemp(
    join(tempRoot ?? tmpdir(), "openklip-url-")
  );
  try {
    let downloadedPath: string;
    try {
      downloadedPath = await downloadVideoFromUrl(
        url,
        temporaryDirectory,
        signal
      );
    } catch (error) {
      throw new Error(safeUrlDownloadErrorMessage(error));
    }
    const result = await ingestDownloadedUrlProject({
      downloadedPath,
      force,
      ingest: ingestProject,
      onProgress,
      signal,
      slug,
    });
    try {
      await buildMediaIndex({
        slug,
        memberProfileId,
        signal,
        onProgress: (status) =>
          onProgress?.({
            phase: "media-index",
            message: status.phase
              ? `Indexing media: ${status.phase}`
              : "Indexing media",
            step: 8,
            total: 8,
          }),
      });
    } catch (error) {
      if (signal?.aborted) {
        throw error;
      }
      const detail = error instanceof Error ? error.message : String(error);
      throw new IngestPartialError(
        "MediaIndexIngestError",
        slug,
        `Project "${slug}" was created, but media indexing failed (${detail}). Rebuild it with media_index_rebuild.`
      );
    }
    return result;
  } finally {
    await rm(temporaryDirectory, { force: true, recursive: true });
  }
}

function assertProjectCanStart(slug: string, force?: boolean): void {
  if (!reserveIngestSlug(slug)) {
    throw new Error(`ingest already in progress for ${slug}`);
  }
  if (!force && existsSync(projectPaths(slug).project)) {
    releaseIngestSlug(slug);
    throw new Error(
      `project already exists: ${slug} (re-ingest would wipe it; set force to overwrite)`
    );
  }
}

/** Start URL creation as an ingest job and return immediately. */
export function startProjectFromUrl(
  input: UrlProjectCreateInput,
  deps: StartUrlProjectDeps = {}
): UrlProjectJobResult {
  const slug = projectSlugFromUrl(input.url, input.projectSlug);
  if (input.memberProfileId) {
    loadMemberProfile(input.memberProfileId);
  }
  assertProjectCanStart(slug, input.force);
  try {
    const job = startIngestJob({
      filename: input.url,
      force: input.force,
      slug,
      sourcePath: input.url,
      run: (onProgress, signal) =>
        (deps.run ?? runUrlProject)({
          ...input,
          onProgress,
          signal,
          slug,
        }).then((result) => result.slug),
    });
    return { jobId: job.id, slug, status: job.status };
  } catch (error) {
    releaseIngestSlug(slug);
    throw error;
  }
}

export function urlProjectJobStatus(jobId: string): IngestJob | null {
  return getIngestJob(jobId) ?? null;
}

/** Agent-facing blocking URL project creation using the browser intake path. */
export async function createProjectFromUrl({
  force,
  memberProfileId,
  projectSlug,
  tempRoot,
  url,
}: UrlProjectCreateInput): Promise<UrlProjectCreateResult> {
  const slug = projectSlugFromUrl(url, projectSlug);
  assertProjectCanStart(slug, force);
  try {
    return await runUrlProject({
      force,
      memberProfileId,
      slug,
      tempRoot,
      url,
    });
  } finally {
    releaseIngestSlug(slug);
  }
}
