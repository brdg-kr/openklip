import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  releaseIngestSlug,
  reserveIngestSlug,
  startIngestJob,
} from "@engine/ingest-jobs";
import { projectPaths } from "@engine/paths";
import {
  downloadVideoFromUrl,
  UrlIngesterUnavailableError,
} from "@engine/url-ingest";
import {
  ingestDownloadedUrlProject,
  projectSlugFromUrl,
  safeUrlDownloadErrorMessage,
  sanitizeUrlProjectFilename,
} from "@engine/url-project";
import type { NextRequest } from "next/server";
import type { IngestFn } from "../post.ts";

export interface UrlProjectsPostDeps {
  loadIngest: () => Promise<IngestFn>;
  tempRoot?: string;
}

export function createUrlProjectsPost({
  loadIngest,
  tempRoot,
}: UrlProjectsPostDeps) {
  return async function POST(req: NextRequest): Promise<Response> {
    const body = (await req.json()) as { url?: string };
    const url = body.url?.trim();
    if (!url) {
      return Response.json({ error: "missing url field" }, { status: 400 });
    }

    const force = new URL(req.url).searchParams.get("force") === "1";
    let slug: string;
    try {
      slug = projectSlugFromUrl(url);
    } catch (error) {
      return Response.json(
        { error: (error as Error).message },
        { status: 400 }
      );
    }
    if (!reserveIngestSlug(slug)) {
      return Response.json(
        {
          code: "in-flight",
          error: `ingest already in progress for ${slug}`,
        },
        { status: 409 }
      );
    }
    if (!force && existsSync(projectPaths(slug).project)) {
      releaseIngestSlug(slug);
      return Response.json(
        {
          code: "exists",
          error: `project already exists: ${slug} (re-ingest would wipe it; confirm to overwrite)`,
        },
        { status: 409 }
      );
    }

    let tmpDir: string;
    try {
      tmpDir = await mkdtemp(join(tempRoot ?? tmpdir(), "openklip-url-"));
    } catch (error) {
      releaseIngestSlug(slug);
      throw error;
    }

    let downloaded: string;
    try {
      downloaded = await downloadVideoFromUrl(url, tmpDir);
    } catch (error) {
      await rm(tmpDir, { recursive: true, force: true });
      releaseIngestSlug(slug);
      if (error instanceof UrlIngesterUnavailableError) {
        return Response.json({ error: error.message }, { status: 503 });
      }
      return Response.json(
        { error: safeUrlDownloadErrorMessage(error) },
        { status: 502 }
      );
    }

    try {
      const filename = sanitizeUrlProjectFilename(downloaded);
      const ingest = await loadIngest();
      const job = startIngestJob({
        filename,
        slug,
        sourcePath: downloaded,
        force,
        run: async (onProgress, signal) => {
          try {
            const result = await ingestDownloadedUrlProject({
              downloadedPath: downloaded,
              force,
              ingest,
              onProgress,
              signal,
              slug,
            });
            return result.slug;
          } finally {
            await rm(tmpDir, { recursive: true, force: true });
          }
        },
      });
      return Response.json({ jobId: job.id, slug });
    } catch (error) {
      releaseIngestSlug(slug);
      await rm(tmpDir, { recursive: true, force: true });
      throw error;
    }
  };
}
