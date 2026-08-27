import { readFile, rename, writeFile } from "node:fs/promises";
import { applyModelEnv, withModelRetry } from "./model-env.mjs";

const [, , command, ...args] = process.argv;
if (command !== "index" && command !== "query" && command !== "queries") {
  process.stderr.write(
    "usage: node media-scene-embed.mjs index <job.json> <out.json> <model> <dtype>\n" +
      "       node media-scene-embed.mjs query <text> <model> <dtype>\n" +
      "       node media-scene-embed.mjs queries <job.json> <model> <dtype>\n"
  );
  process.exit(2);
}

const {
  env,
  AutoProcessor,
  AutoTokenizer,
  RawImage,
  SiglipTextModel,
  SiglipVisionModel,
} = await import("@huggingface/transformers");
applyModelEnv(env);

function normalize(values) {
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

const load = (operation) =>
  withModelRetry(operation, {
    onRetry: (error, attempt, delay) =>
      process.stderr.write(
        `[media-scene] model load failed on attempt ${attempt}, retrying in ${delay}ms: ${error?.message ?? error}\n`
      ),
  });

if (command === "query" || command === "queries") {
  const [input, model, dtype = "int8"] = args;
  if (!(input && model)) {
    throw new Error(`${command} requires input and model`);
  }
  const texts =
    command === "query"
      ? [input]
      : JSON.parse(await readFile(input, "utf8")).queries;
  if (
    !Array.isArray(texts) ||
    texts.length < 1 ||
    texts.length > 3 ||
    texts.some((text) => typeof text !== "string" || !text.trim())
  ) {
    throw new Error("queries must contain 1-3 non-empty strings");
  }
  const tokenizer = await load(() => AutoTokenizer.from_pretrained(model));
  const textModel = await load(() =>
    SiglipTextModel.from_pretrained(model, { dtype })
  );
  const inputs = tokenizer(texts, {
    max_length: 64,
    padding: "max_length",
    truncation: true,
  });
  const output = await textModel(inputs);
  const data = output.pooler_output.data;
  const dim = output.pooler_output.dims.at(-1);
  const vectors = texts.map((text, index) => {
    const vector = normalize(
      Float32Array.from(data.slice(index * dim, (index + 1) * dim))
    );
    return { text, vector: Array.from(vector) };
  });
  process.stdout.write(
    `${JSON.stringify({
      dim,
      dtype,
      model,
      vectors,
      ...(command === "query" ? { vector: vectors[0].vector } : {}),
    })}\n`
  );
  process.exit(0);
}

const [jobPath, outputPath, model, dtype = "int8"] = args;
if (!(jobPath && outputPath && model)) {
  throw new Error("index requires job, output, and model");
}
const job = JSON.parse(await readFile(jobPath, "utf8"));
const processor = await load(() => AutoProcessor.from_pretrained(model));
const visionModel = await load(() =>
  SiglipVisionModel.from_pretrained(model, { dtype })
);
const images = [];
for (let index = 0; index < job.images.length; index += 1) {
  const item = job.images[index];
  const image = await RawImage.read(item.path);
  const output = await visionModel(await processor(image));
  const vector = normalize(Float32Array.from(output.pooler_output.data));
  images.push({
    ...item,
    vectorB64: Buffer.from(
      vector.buffer,
      vector.byteOffset,
      vector.byteLength
    ).toString("base64"),
  });
  if ((index + 1) % 25 === 0 || index === job.images.length - 1) {
    process.stderr.write(
      `[media-scene] ${index + 1}/${job.images.length} image(s)\n`
    );
  }
}
const temporary = `${outputPath}.${process.pid}.tmp`;
await writeFile(
  temporary,
  JSON.stringify({
    dim: images[0]?.vectorB64 ? 768 : 0,
    dtype,
    images,
    model,
    version: 1,
  })
);
await rename(temporary, outputPath);
