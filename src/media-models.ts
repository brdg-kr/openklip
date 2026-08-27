import { createHash } from "node:crypto";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { stateDir } from "./repo-paths.ts";

export const FACE_DETECTOR_MODEL = {
  id: "opencv-yunet-2023mar-fp32",
  filename: "face_detection_yunet_2023mar.onnx",
  sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
  url: "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
} as const;

export const FACE_RECOGNIZER_MODEL = {
  id: "opencv-sface-2021dec-fp32",
  filename: "face_recognition_sface_2021dec.onnx",
  sha256: "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
  url: "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
} as const;

export const SCENE_EMBEDDING_MODEL = {
  dim: 768,
  dtype: "int8",
  id: "onnx-community/siglip2-base-patch16-224-ONNX",
} as const;

interface DownloadableModel {
  filename: string;
  id: string;
  sha256: string;
  url: string;
}

export interface FaceModelPaths {
  detector: string;
  recognizer: string;
}

function mediaModelRoot(): string {
  const configured = process.env.OPENKLIP_MODEL_CACHE?.trim();
  return configured ? join(configured, "openklip-media") : stateDir("models");
}

function modelPath(model: DownloadableModel): string {
  return join(mediaModelRoot(), model.id, model.filename);
}

async function fileSha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return hash.digest("hex");
}

function offlineRequested(): boolean {
  return (
    process.env.TRANSFORMERS_OFFLINE === "1" ||
    process.env.HF_HUB_OFFLINE === "1"
  );
}

async function ensureModel(
  model: DownloadableModel,
  fetchImpl: typeof fetch
): Promise<string> {
  const destination = modelPath(model);
  if (existsSync(destination)) {
    const existingHash = await fileSha256(destination);
    if (existingHash === model.sha256) {
      return destination;
    }
    await rm(destination, { force: true });
  }
  if (offlineRequested()) {
    throw new Error(`media model is not cached for offline use: ${model.id}`);
  }
  await mkdir(dirname(destination), { recursive: true });
  const response = await fetchImpl(model.url, { redirect: "follow" });
  if (!(response.ok && response.body)) {
    throw new Error(
      `could not download media model ${model.id}: HTTP ${response.status}`
    );
  }
  const temporary = `${destination}.${process.pid}.tmp`;
  try {
    const bytes = new Uint8Array(await response.arrayBuffer());
    await writeFile(temporary, bytes);
    const downloadedHash = await fileSha256(temporary);
    if (downloadedHash !== model.sha256) {
      throw new Error(
        `media model checksum mismatch for ${model.id}: expected ${model.sha256}, got ${downloadedHash}`
      );
    }
    await rename(temporary, destination);
  } finally {
    await rm(temporary, { force: true });
  }
  return destination;
}

export async function ensureFaceModels(
  fetchImpl: typeof fetch = fetch
): Promise<FaceModelPaths> {
  const [detector, recognizer] = await Promise.all([
    ensureModel(FACE_DETECTOR_MODEL, fetchImpl),
    ensureModel(FACE_RECOGNIZER_MODEL, fetchImpl),
  ]);
  return { detector, recognizer };
}
