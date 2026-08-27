import assert from "node:assert/strict";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { projectPaths } from "../src/paths.ts";
import { resetIngestJobsForTests } from "../src/ingest-jobs.ts";
import {
  ingestDownloadedUrlProject,
  projectSlugFromUrl,
  startProjectFromUrl,
  urlProjectJobStatus,
} from "../src/url-project.ts";
import {
  makeProject,
  withTempProjectsRoot,
  writeFixtureProject,
} from "./helpers/projectFixture.ts";

test("projectSlugFromUrl uses the case-sensitive YouTube video id", () => {
  const id = "AbC_dEf-123";
  assert.equal(
    projectSlugFromUrl(`https://www.youtube.com/watch?v=${id}&t=3`),
    `youtube-${id}`
  );
  assert.equal(projectSlugFromUrl(`https://youtu.be/${id}`), `youtube-${id}`);
  assert.equal(
    projectSlugFromUrl(`https://www.youtube.com/shorts/${id}`),
    `youtube-${id}`
  );
});

test("projectSlugFromUrl makes generic URL slugs stable and collision-resistant", () => {
  const first = projectSlugFromUrl("https://media.example/videos/episode.mp4");
  const repeat = projectSlugFromUrl("https://media.example/videos/episode.mp4");
  const other = projectSlugFromUrl("https://other.example/videos/episode.mp4");
  assert.equal(first, repeat);
  assert.notEqual(first, other);
  assert.match(first, /^url-media-example-episode-mp4-[a-f0-9]{10}$/);
  assert.ok(first.length <= 64);
});

test("projectSlugFromUrl validates explicit slugs and rejects credentialed URLs", () => {
  assert.equal(
    projectSlugFromUrl("https://example.com/video", "my-project"),
    "my-project"
  );
  assert.throws(
    () => projectSlugFromUrl("https://example.com/video", "../escape"),
    /invalid project slug/
  );
  assert.throws(
    () => projectSlugFromUrl("https://user:secret@example.com/video"),
    /embedded credentials/
  );
});

test("ingestDownloadedUrlProject passes the requested slug and persists the source", async () => {
  await withTempProjectsRoot(async ({ root }) => {
    const slug = "youtube-AbC_dEf-123";
    const downloadedPath = join(root, "download.mp4");
    writeFileSync(downloadedPath, "downloaded-source-bytes");

    const result = await ingestDownloadedUrlProject({
      downloadedPath,
      slug,
      ingest: (videoPath, opts) => {
        assert.equal(videoPath, downloadedPath);
        assert.equal(opts?.slug, slug);
        writeFixtureProject(
          slug,
          makeProject({ slug, source: downloadedPath })
        );
        return Promise.resolve(slug);
      },
    });

    assert.equal(result.slug, slug);
    assert.equal(result.filename, "download.mp4");
    assert.equal(
      readFileSync(result.sourcePath, "utf8"),
      "downloaded-source-bytes"
    );
    assert.equal(existsSync(projectPaths(slug).project), true);
    const project = JSON.parse(
      readFileSync(projectPaths(slug).project, "utf8")
    ) as { source: string };
    assert.equal(project.source, result.sourcePath);
  });
});

test("startProjectFromUrl returns immediately and exposes completion through job status", async () => {
  await withTempProjectsRoot(async () => {
    resetIngestJobsForTests();
    const slug = "youtube-AbC_dEf-123";
    const started = startProjectFromUrl(
      {
        url: "https://youtu.be/AbC_dEf-123",
      },
      {
        run: () =>
          Promise.resolve({
            filename: "download.mp4",
            projectDir: projectPaths(slug).dir,
            slug,
            sourcePath: join(projectPaths(slug).dir, "download.mp4"),
          }),
      }
    );
    assert.equal(started.slug, slug);
    assert.equal(started.status, "running");

    let job = urlProjectJobStatus(started.jobId);
    for (
      let attempt = 0;
      attempt < 20 && job?.status === "running";
      attempt++
    ) {
      await new Promise((resolve) => setTimeout(resolve, 5));
      job = urlProjectJobStatus(started.jobId);
    }
    assert.equal(job?.status, "done");
    resetIngestJobsForTests();
  });
});
