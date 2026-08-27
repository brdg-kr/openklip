import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mock, test } from "bun:test";
import { withTempProjectsRoot } from "./helpers/projectFixture.ts";

const realUrlIngest = await import("../src/url-ingest.ts");

mock.module("@engine/url-ingest", () => ({
  UrlIngesterUnavailableError: realUrlIngest.UrlIngesterUnavailableError,
  downloadVideoFromUrl: () =>
    Promise.reject(
      new Error(
        "yt-dlp failed (exit 1): HTTP Error 403: https://media.example/video?sig=fake"
      )
    ),
}));

const { createUrlProjectsPost } = await import(
  "../app/api/projects/url/post.ts"
);

function urlRequest(url: string) {
  return new Request("http://localhost/api/projects/url", {
    body: JSON.stringify({ url }),
    headers: { "content-type": "application/json" },
    method: "POST",
  }) as unknown as Parameters<ReturnType<typeof createUrlProjectsPost>>[0];
}

test("POST /api/projects/url returns a safe JSON error when yt-dlp fails", async () => {
  await withTempProjectsRoot(async () => {
    const tempRoot = await mkdtemp(join(tmpdir(), "openklip-url-route-test-"));
    try {
      const post = createUrlProjectsPost({
        loadIngest: () => Promise.reject(new Error("ingest must not start")),
        tempRoot,
      });

      const response = await post(urlRequest("https://youtu.be/example"));
      assert.equal(response.status, 502);
      assert.match(response.headers.get("content-type") ?? "", /json/i);

      const json = (await response.json()) as { error?: string };
      assert.match(json.error ?? "", /HTTP 403/i);
      assert.doesNotMatch(json.error ?? "", /media\.example|sig=/i);
    } finally {
      await rm(tempRoot, { force: true, recursive: true });
    }
  });
});
