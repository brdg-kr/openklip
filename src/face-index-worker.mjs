import { readFile, rename, writeFile } from "node:fs/promises";
import * as ort from "onnxruntime-node";
import sharp from "sharp";

const [, , jobPath, outputPath] = process.argv;
if (!(jobPath && outputPath)) {
  process.stderr.write(
    "usage: node face-index-worker.mjs <job.json> <output.json>\n"
  );
  process.exit(2);
}

const DETECTOR_SIZE = 640;
const DETECTOR_STRIDES = [8, 16, 32];
const DETECTION_THRESHOLD = 0.82;
const NMS_THRESHOLD = 0.3;
const FACE_SIZE = 112;
const SFACE_TARGETS = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toNchwBgr(raw, width, height) {
  const plane = width * height;
  const tensor = new Float32Array(plane * 3);
  for (let pixel = 0; pixel < plane; pixel += 1) {
    const rawOffset = pixel * 3;
    tensor[pixel] = raw[rawOffset + 2];
    tensor[plane + pixel] = raw[rawOffset + 1];
    tensor[plane * 2 + pixel] = raw[rawOffset];
  }
  return tensor;
}

function toNchwRgb(raw, width, height) {
  const plane = width * height;
  const tensor = new Float32Array(plane * 3);
  for (let pixel = 0; pixel < plane; pixel += 1) {
    const rawOffset = pixel * 3;
    tensor[pixel] = raw[rawOffset];
    tensor[plane + pixel] = raw[rawOffset + 1];
    tensor[plane * 2 + pixel] = raw[rawOffset + 2];
  }
  return tensor;
}

function intersectionOverUnion(left, right) {
  const x1 = Math.max(left.x, right.x);
  const y1 = Math.max(left.y, right.y);
  const x2 = Math.min(left.x + left.width, right.x + right.width);
  const y2 = Math.min(left.y + left.height, right.y + right.height);
  const width = Math.max(0, x2 - x1);
  const height = Math.max(0, y2 - y1);
  const intersection = width * height;
  const union =
    left.width * left.height + right.width * right.height - intersection;
  return union > 0 ? intersection / union : 0;
}

function nonMaximumSuppression(faces) {
  const sorted = [...faces].sort(
    (left, right) => right.confidence - left.confidence
  );
  const kept = [];
  for (const candidate of sorted) {
    if (
      kept.every(
        (existing) =>
          intersectionOverUnion(candidate.box, existing.box) < NMS_THRESHOLD
      )
    ) {
      kept.push(candidate);
    }
  }
  return kept;
}

function decodeDetectorOutputs(outputs, originalWidth, originalHeight) {
  const scaleX = originalWidth / DETECTOR_SIZE;
  const scaleY = originalHeight / DETECTOR_SIZE;
  const faces = [];
  for (const stride of DETECTOR_STRIDES) {
    const columns = DETECTOR_SIZE / stride;
    const rows = DETECTOR_SIZE / stride;
    const cls = outputs[`cls_${stride}`].data;
    const obj = outputs[`obj_${stride}`].data;
    const bbox = outputs[`bbox_${stride}`].data;
    const keypoints = outputs[`kps_${stride}`].data;
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const index = row * columns + column;
        const clsScore = clamp(cls[index], 0, 1);
        const objScore = clamp(obj[index], 0, 1);
        const confidence = Math.sqrt(clsScore * objScore);
        if (confidence < DETECTION_THRESHOLD) {
          continue;
        }
        const centerX = (column + bbox[index * 4]) * stride;
        const centerY = (row + bbox[index * 4 + 1]) * stride;
        const width = Math.exp(bbox[index * 4 + 2]) * stride;
        const height = Math.exp(bbox[index * 4 + 3]) * stride;
        const landmarks = [];
        for (let point = 0; point < 5; point += 1) {
          landmarks.push([
            (keypoints[index * 10 + point * 2] + column) * stride * scaleX,
            (keypoints[index * 10 + point * 2 + 1] + row) * stride * scaleY,
          ]);
        }
        const x = clamp((centerX - width / 2) * scaleX, 0, originalWidth);
        const y = clamp((centerY - height / 2) * scaleY, 0, originalHeight);
        const mappedWidth = clamp(width * scaleX, 0, originalWidth - x);
        const mappedHeight = clamp(height * scaleY, 0, originalHeight - y);
        faces.push({
          box: { x, y, width: mappedWidth, height: mappedHeight },
          confidence,
          landmarks,
        });
      }
    }
  }
  return nonMaximumSuppression(faces);
}

function similarityTransform(source, target) {
  const sourceMean = source.reduce(
    (sum, point) => [sum[0] + point[0], sum[1] + point[1]],
    [0, 0]
  );
  const targetMean = target.reduce(
    (sum, point) => [sum[0] + point[0], sum[1] + point[1]],
    [0, 0]
  );
  sourceMean[0] /= source.length;
  sourceMean[1] /= source.length;
  targetMean[0] /= target.length;
  targetMean[1] /= target.length;
  let denominator = 0;
  let real = 0;
  let imaginary = 0;
  for (let index = 0; index < source.length; index += 1) {
    const sourceX = source[index][0] - sourceMean[0];
    const sourceY = source[index][1] - sourceMean[1];
    const targetX = target[index][0] - targetMean[0];
    const targetY = target[index][1] - targetMean[1];
    denominator += sourceX * sourceX + sourceY * sourceY;
    real += sourceX * targetX + sourceY * targetY;
    imaginary += sourceX * targetY - sourceY * targetX;
  }
  if (!(denominator > 0)) {
    throw new Error("face landmarks do not define a valid transform");
  }
  const a = real / denominator;
  const b = imaginary / denominator;
  return {
    a,
    b,
    tx: targetMean[0] - a * sourceMean[0] + b * sourceMean[1],
    ty: targetMean[1] - b * sourceMean[0] - a * sourceMean[1],
  };
}

function sampleBilinear(raw, width, height, x, y, channel) {
  if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
    return 0;
  }
  const left = Math.floor(x);
  const top = Math.floor(y);
  const right = Math.min(width - 1, left + 1);
  const bottom = Math.min(height - 1, top + 1);
  const xWeight = x - left;
  const yWeight = y - top;
  const at = (column, row) => raw[(row * width + column) * 3 + channel];
  const topValue = at(left, top) * (1 - xWeight) + at(right, top) * xWeight;
  const bottomValue =
    at(left, bottom) * (1 - xWeight) + at(right, bottom) * xWeight;
  return topValue * (1 - yWeight) + bottomValue * yWeight;
}

function alignFace(raw, width, height, landmarks) {
  const transform = similarityTransform(landmarks, SFACE_TARGETS);
  const determinant = transform.a * transform.a + transform.b * transform.b;
  const aligned = new Uint8Array(FACE_SIZE * FACE_SIZE * 3);
  for (let targetY = 0; targetY < FACE_SIZE; targetY += 1) {
    for (let targetX = 0; targetX < FACE_SIZE; targetX += 1) {
      const translatedX = targetX - transform.tx;
      const translatedY = targetY - transform.ty;
      const sourceX =
        (transform.a * translatedX + transform.b * translatedY) / determinant;
      const sourceY =
        (-transform.b * translatedX + transform.a * translatedY) / determinant;
      const offset = (targetY * FACE_SIZE + targetX) * 3;
      for (let channel = 0; channel < 3; channel += 1) {
        aligned[offset + channel] = Math.round(
          sampleBilinear(raw, width, height, sourceX, sourceY, channel)
        );
      }
    }
  }
  return aligned;
}

function faceSharpness(raw) {
  const values = [];
  for (let y = 1; y < FACE_SIZE - 1; y += 1) {
    for (let x = 1; x < FACE_SIZE - 1; x += 1) {
      const gray = (column, row) => {
        const offset = (row * FACE_SIZE + column) * 3;
        return (
          raw[offset] * 0.299 +
          raw[offset + 1] * 0.587 +
          raw[offset + 2] * 0.114
        );
      };
      values.push(
        gray(x, y) * 4 -
          gray(x - 1, y) -
          gray(x + 1, y) -
          gray(x, y - 1) -
          gray(x, y + 1)
      );
    }
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return (
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length
  );
}

function normalizeVector(values) {
  let sumSquares = 0;
  for (const value of values) {
    sumSquares += value * value;
  }
  const norm = Math.sqrt(sumSquares) || 1;
  const normalized = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    normalized[index] = values[index] / norm;
  }
  return normalized;
}

async function analyzeImage(detector, recognizer, item) {
  const source = sharp(item.path).removeAlpha();
  const { data: raw, info } = await source
    .clone()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const detectorRaw = await source
    .resize(DETECTOR_SIZE, DETECTOR_SIZE, { fit: "fill" })
    .raw()
    .toBuffer();
  const detectorInput = new ort.Tensor(
    "float32",
    toNchwBgr(detectorRaw, DETECTOR_SIZE, DETECTOR_SIZE),
    [1, 3, DETECTOR_SIZE, DETECTOR_SIZE]
  );
  const detectorOutput = await detector.run({ input: detectorInput });
  const faces = decodeDetectorOutputs(detectorOutput, info.width, info.height);
  const analyzedFaces = [];
  for (const face of faces) {
    const aligned = alignFace(raw, info.width, info.height, face.landmarks);
    const recognitionInput = new ort.Tensor(
      "float32",
      toNchwRgb(aligned, FACE_SIZE, FACE_SIZE),
      [1, 3, FACE_SIZE, FACE_SIZE]
    );
    const recognitionOutput = await recognizer.run({ data: recognitionInput });
    const vector = normalizeVector(recognitionOutput.fc1.data);
    analyzedFaces.push({
      ...face,
      sharpness: faceSharpness(aligned),
      vectorB64: Buffer.from(
        vector.buffer,
        vector.byteOffset,
        vector.byteLength
      ).toString("base64"),
    });
  }
  return {
    atSec: item.atSec ?? null,
    height: info.height,
    id: item.id,
    path: item.path,
    width: info.width,
    faces: analyzedFaces,
  };
}

const job = JSON.parse(await readFile(jobPath, "utf8"));
const detector = await ort.InferenceSession.create(job.models.detector, {
  logSeverityLevel: 4,
});
const recognizer = await ort.InferenceSession.create(job.models.recognizer, {
  logSeverityLevel: 4,
});
const images = [];
for (let index = 0; index < job.images.length; index += 1) {
  images.push(await analyzeImage(detector, recognizer, job.images[index]));
  if ((index + 1) % 100 === 0 || index === job.images.length - 1) {
    process.stderr.write(
      `[face-index] ${index + 1}/${job.images.length} image(s)\n`
    );
  }
}
const temporary = `${outputPath}.${process.pid}.tmp`;
await writeFile(
  temporary,
  JSON.stringify({
    detectorModel: job.detectorModel,
    images,
    recognizerModel: job.recognizerModel,
    version: 1,
  })
);
await rename(temporary, outputPath);
