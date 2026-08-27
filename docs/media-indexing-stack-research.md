# Media indexing stack research

Research snapshot: 2026-08-28

Scope: low-cost, high-precision local indexing and retrieval for FanTube source
videos. This document records research evidence, evidence limits, and the query
contract used to evaluate the selected stack. It does not schedule future work
or prove that an unmeasured quality target has been met.

## Selected architecture

Identity and scene meaning remain separate indexes joined on source time:

1. The face identity index answers who appears and when.
2. The scene meaning index answers what is visibly happening in the whole shot
   and around an identified target member.

This separation is also the language boundary. A Korean member name is resolved
to a `memberProfileId`; it is not compared with an image embedding. Natural
language is used only for the scene, transcript, OCR, and other semantic parts
of a request.

The selected local stack in the current source is:

| Layer | Selected component | Evidence and boundary |
| --- | --- | --- |
| Face detection | [OpenCV YuNet FP32 ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Small local detector with five landmarks. OpenCV Zoo publishes WIDER Face results and an MIT license for this model directory. |
| Face recognition | [OpenCV SFace FP32 ONNX](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Local 128-dimensional face embeddings. OpenCV Zoo publishes verification results and an Apache-2.0 license for this model directory. |
| Scene embedding | [`onnx-community/siglip2-base-patch16-224-ONNX`](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX), INT8 | Transformers.js-compatible conversion of Google SigLIP2 Base. The conversion has no separately published Korean or quantization-fidelity benchmark. |
| Scene source model | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | Apache-2.0, 224 pixel input, 768-dimensional joint image-text space. Google identifies image-text retrieval as an intended use. |
| Per-project storage and search | Model-versioned vectors in `working/media-index.json`, exact normalized dot product | Derived data stays outside `project.json`. Exact search avoids approximate-nearest-neighbor recall loss at one-video scale. |

The older `working/moment-index.json` is a separate broad visual-search index.
It samples the proxy every three seconds and uses
`Xenova/clip-vit-base-patch32`. It remains useful as a regression baseline but
does not identify a member.

## Face identity evidence

### Candidate comparison

| Candidate | Owner-published quality signal | Local and licensing boundary |
| --- | --- | --- |
| [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | WIDER Face AP 0.8844 easy, 0.8656 medium, 0.7503 hard for the current FP model | 233 KB ONNX detector; model directory is MIT. |
| [OpenCV SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | OpenCV Zoo reports 0.9940 verification accuracy for FP, 0.9942 block-quantized, and 0.9932 INT8 | 38.7 MB FP32 recognizer, 112 x 112 aligned input, 128-dimensional output; model directory is Apache-2.0. |
| [InsightFace `buffalo_l`](https://github.com/deepinsight/insightface/blob/master/python-package/README.md) | Owner model zoo reports LFW 99.83, CFP-FP 99.33, AgeDB-30 98.23, and IJB-C E4 97.25 | Public weights are restricted to non-commercial research. They are an evaluation reference, not a shippable default without a commercial license. |
| [CVLFace AdaFace IR101 WebFace12M](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m) | The [owner board](https://github.com/mk-minchul/CVLface) reports LFW 99.82, CFP-FP 99.24, AgeDB 98.00, IJB-C at 0.01 97.72, and TinyFace rank-1 72.42 | Repository code is MIT, while the model card directs users to the training-dataset license. No owner-published drop-in ONNX artifact is the primary path. |

These figures use different datasets and protocols, so they are quality signals
only. They do not establish idol-video accuracy. In-video identity decisions
must use aligned face crops, multiple target references, same-group hard
negatives, absolute similarity, runner-up margin, and track-level evidence.
`UNKNOWN` and `AMBIGUOUS` are valid results; the system must not force every
face into a member identity.

## SigLIP2 multilingual evidence

### What the primary sources establish

The [SigLIP2 paper](https://arxiv.org/abs/2502.14786) states that the model uses
the multilingual Gemma tokenizer with a 256,000-token vocabulary. Training uses
WebLI, which covers 109 languages. The training mixture is 90% image-text pairs
from English web pages and 10% from non-English web pages.

Google evaluates multilingual image-text retrieval on Crossmodal-3600
(XM3600), a dataset of 3,600 images with human-written captions in 36 languages,
including Korean. For the exact B/16 224 source checkpoint, Table 1 reports
average XM3600 Recall@1 of 40.3 for text-to-image retrieval and 50.7 for
image-to-text retrieval. The paper's per-language Figure 2 includes Korean and
shows direct Korean retrieval that is not below English in the plotted
comparison. However, the figure does not publish numeric per-language rows or
identify its Korean bars as results from the B/16 224 checkpoint.

The [Google checkpoint README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md)
reports B/16 224 English-focused COCO Recall@1 of 52.1 for text-to-image and
68.9 for image-to-text retrieval. Those COCO numbers and the XM3600 averages
come from different datasets and cannot be used to calculate a Korean-versus-
English quality delta.

The [official Transformers SigLIP2 documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md)
requires the training-time text preprocessing: lowercase text, fixed padding,
truncation, and a maximum length of 64 tokens. The shipped tokenizer uses byte
fallback, so Korean text can be encoded rather than becoming unknown text.

### What is not established

The available primary sources do not establish any of the following:

- Korean and English retrieval quality are equal for
  `google/siglip2-base-patch16-224`.
- The ONNX Community INT8 conversion preserves the source checkpoint's Korean
  retrieval quality within a known tolerance.
- XM3600 caption retrieval predicts quality for short, colloquial Korean idol
  queries such as reactions, gestures, props, or broadcast-stage actions.
- Translating a Korean request to English always improves retrieval.
- Adding several paraphrases always improves top-result precision.

Therefore Korean is a supported direct query language, not a proven
quality-equivalent language for this exact runtime artifact. Translation is a
low-cost recall hedge that remains subject to in-domain measurement.

## Agent scene-query contract

### Input separation

The Agent converts a Korean user request into identity and scene constraints:

```json
{
  "memberProfileId": "nmixx-haewon",
  "sceneQueries": [
    {
      "kind": "korean-original",
      "text": "무대에서 갑자기 놀라 크게 웃는 사람"
    },
    {
      "kind": "english-translation",
      "text": "a performer suddenly looks surprised and laughs loudly on stage"
    },
    {
      "kind": "visual-attributes",
      "text": "surprised expression, open-mouth laugh, stage close-up"
    }
  ]
}
```

The member name is resolved to `memberProfileId` and removed from the visual
description. Face matching supplies identity. Scene queries describe visible
evidence only.

### Rewrite rules

- Preserve the Korean scene clause verbatim after removing the resolved member
  name and non-visual instructions.
- Produce at most one faithful English translation. Preserve action, object,
  expression, setting, direction, and negation. Do not add an event that the
  user did not request.
- Produce at most one compact visual-attribute query when it adds observable
  cues that a frame can contain. Do not add personality, intent, popularity,
  or other invisible judgments.
- Phrase each variant as a short image caption rather than a question. Keep it
  below the model's 64-token limit.
- Use only the Korean original when translation or visual expansion adds no
  distinct retrieval signal.
- Skip scene embedding entirely when the request asks only when the member
  appears. Return the face appearance timeline instead.

This contract caps one user request at three text embeddings. It does not
re-embed video frames, alter the stored index, or call an Agent for each frame.
Normalized query vectors can be cached by model id, dtype, preprocessing
version, and query text.

### Ranking contract

1. Filter scene rows by the target member's verified appearance intervals.
2. Preserve the whole-frame context rows and the member-centered crop rows that
   overlap those intervals.
3. Embed all query variants in one text-model batch using the exact
   training-time preprocessing.
4. Compute exact cosine similarity against every eligible stored scene vector
   for each query variant.
5. Fuse the per-query ranks with deterministic reciprocal-rank fusion. Do not
   average raw cosine values across languages unless an in-domain calibration
   shows their score distributions are comparable.
6. Return the fused rank together with each query kind's exact cosine score,
   source time, crop scope, and member-appearance evidence.

The Korean original always participates in fusion. English is not a hidden
replacement for the user's meaning. Exposing the query variants and their
individual scores makes translation drift reviewable.

## Bilingual retrieval acceptance contract

The evaluation unit is a labeled set of real Korean FanTube requests paired
with the relevant source-time intervals. Each request has a human-checked
English translation. Identity filtering, indexed image vectors, candidate
spans, and relevance labels stay fixed while comparing these query policies:

1. Korean original only.
2. English translation only.
3. Korean plus English rank fusion.
4. Korean plus English plus visual-attribute rank fusion.

Report Recall@5, Recall@20, precision@5, nDCG@10, mean reciprocal rank, and
temporal intersection-over-union. Report results separately for expression,
gesture, action, prop, setting, and multi-member attribution queries. Use paired
bootstrap confidence intervals so a small corpus does not turn noise into a
language policy.

The decision rules are:

- Korean-only remains the cheapest default when its Recall@5 and nDCG@10 are
  within 2 percentage points of the better single-language policy and fusion
  improves neither metric by at least 2 percentage points.
- Bilingual fusion is enabled when it improves Recall@5 or nDCG@10 by at least
  2 percentage points without reducing precision@5 by more than 1 percentage
  point.
- The visual-attribute variant is retained only when it adds the same measured
  benefit over bilingual fusion.
- INT8 remains acceptable only when its paired retrieval metrics are within
  1 percentage point of the source or higher-precision runtime on the same
  queries and vectors.
- Multi-query execution adds no image inference or index storage. Its warm
  text-embedding and exact-ranking p95 is reported relative to Korean-only so
  quality gains and query cost remain visible together.

These are evaluation rules, not measured results. Until paired Korean idol
retrieval results exist, neither Korean-only nor translation-first can be
described as the higher-quality policy.

## Index quality measurements

Public face and scene benchmarks do not replace in-domain measurements. The
same acceptance corpus reports:

| Area | Measurements |
| --- | --- |
| Face identity | Frame precision and recall, track precision and recall, same-group false-positive rate, `UNKNOWN` rate |
| Time | Appearance-boundary error and recall for intervals shorter than three seconds |
| Scene retrieval | Recall@5, Recall@20, precision@5, nDCG@10, mean reciprocal rank, temporal intersection-over-union |
| Member attribution | Whole-frame versus member-centered retrieval on multi-member shots |
| Runtime cost | Cold model download, peak RSS, indexing wall time and disk per source minute, warm query p95 |

All model, dtype, preprocessing, source, profile, and reference-image hashes are
part of the evidence key. Results from a different model size, precision,
source file, or member profile do not validate the current index.

## Primary sources

- SigLIP2 paper, including WebLI language mixture, B/16 224 XM3600 results, and
  Korean in the per-language retrieval figure:
  <https://arxiv.org/abs/2502.14786>
- Google SigLIP2 checkpoint README and published checkpoint results:
  <https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md>
- Google SigLIP2 Base model card and intended uses:
  <https://huggingface.co/google/siglip2-base-patch16-224>
- Transformers.js-compatible ONNX conversion used by OpenKlip:
  <https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX>
- Official Transformers SigLIP2 preprocessing and retrieval documentation:
  <https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md>
- Official Transformers SigLIP2 tokenizer implementation:
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/tokenization_siglip2.py>
- Crossmodal-3600 paper and dataset definition:
  <https://arxiv.org/abs/2205.12522>
- OpenCV Zoo YuNet model card and artifacts:
  <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- OpenCV Zoo SFace model card and artifacts:
  <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>
- OpenCV SFace alignment and preprocessing implementation:
  <https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp>
- InsightFace model zoo and public-weight license terms:
  <https://github.com/deepinsight/insightface/blob/master/python-package/README.md>
- CVLFace owner model board:
  <https://github.com/mk-minchul/CVLface>
