import { copyFile, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { projectPaths } from "./paths.ts";
import { withProjectLock } from "./project-lock.ts";

/**
 * Persist a temporary ingest source beside project.json and repoint the EDL.
 *
 * Upload, folder, and URL ingest all start from temporary files that disappear
 * after the request or background job settles. Export needs the full-resolution
 * source to remain available, so every intake surface uses this one function.
 */
export async function persistProjectSource(
  slug: string,
  filename: string,
  temporaryPath: string
): Promise<string> {
  const paths = projectPaths(slug);
  const storedSource = join(paths.dir, filename);
  await copyFile(temporaryPath, storedSource);
  // This internal repoint is part of ingest finalization. It must serialize
  // with editor saves but must not create a user-facing edit revision.
  await withProjectLock(slug, async () => {
    const project = JSON.parse(await readFile(paths.project, "utf8")) as {
      source?: string;
    };
    project.source = storedSource;
    await writeFile(paths.project, JSON.stringify(project, null, 2));
  });
  return storedSource;
}
