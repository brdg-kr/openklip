---
name: target-moment-search
description: Search an indexed source for one target member using Korean input and bounded bilingual scene-query fusion
---

# Target moment search

Use this playbook after a target member profile and project media index are ready.

The user may speak only Korean. Keep that as the authoritative intent. Face identity is never translated or inferred from scene text: resolve it to one exact `memberProfileId`.

## Preconditions

1. Call `member_profile_list` and resolve the exact target profile.
2. Call `media_index_status`.
3. If the index is missing, stale, interrupted, or built for another profile, call `media_index_rebuild` and poll until `done`.
4. Use `member_appearance_search` to confirm that matched or ambiguous intervals exist.

## Build at most three scene queries

Always keep the Korean request as query 1.

Add no more than two variants:

1. A faithful English visual translation. Preserve actions, objects, expression, setting, and negation. Remove the member name because identity is handled by `memberProfileId`.
2. An optional visible-attribute English caption only when it adds concrete visual cues, such as close-up, profile view, holding a microphone, laughing, or surprised expression.

Do not invent dialogue, motive, chronology, popularity, or hidden emotion. Do not replace the Korean query with English.

Example:

- Korean: `해원이 갑자기 놀라서 크게 웃는 장면`
- English: `a woman suddenly laughing after a surprised reaction`
- Visible attributes: `close-up surprised expression followed by a wide smile and laughter`

## Search

Call `moment_search` with:

- `slug`
- original Korean `query`
- `queryVariants` containing zero to two English variants
- exact `memberProfileId`
- a bounded `limit`, normally 12

The tool embeds all variants in one warm model process, performs exact cosine scoring per variant, applies reciprocal-rank fusion, clusters adjacent seconds, and returns per-query scores.

## Judge results

- Prefer spans supported by the face timeline and at least two query variants.
- Keep Korean-only results when the Korean score and rank are already strong.
- Treat `ambiguous` face intervals as review candidates, never confirmed identity.
- Inspect the top thumbnails and surrounding source window before proposing a topic or edit range.
- Report which query variant found each candidate. Do not hide translation drift.

The index retrieves candidates. The Agent chooses the topic and final edit window.

