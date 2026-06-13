---
title: DreamVoice
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: true
tags:
  - thousand-token-wood
  - openbmb
  - minicpm
  - voxcpm2
  - voice-cloning
  - bedtime-stories
---

# 🌙 DreamVoice — Bedtime stories in Mom's own voice

> Mom is traveling for work. Her 6-year-old can't sleep. She opens DreamVoice, taps
> **🐉 Dragons + 😄 Funny**, and hears *her mom's voice* — cloned, warm, familiar — tell a
> story about a clumsy dragon who sneezes glitter. She's asleep in three minutes, smiling.

DreamVoice clones a parent's voice from a short recording and narrates a custom, AI-generated
bedtime story in it — so a child hears a loved one read to them, even when they're far away.

**Built for the Build Small Hackathon (Hugging Face × Gradio, June 2026) — Thousand Token Wood track.**

## How it works
1. A parent records 30–60s of their voice (or uploads a clip).
2. **`openbmb/VoxCPM2`** clones the voice.
3. The child picks a **genre** (animals, kingdom, space, dragons, ocean, forest) and **mood**
   (magical, funny, calming, dreamy), and optionally names the hero.
4. **`openbmb/MiniCPM5-1B`** writes a ~150-word, age-6, calming bedtime story.
5. VoxCPM2 narrates the story **in the parent's cloned voice**, ending on a gentle goodnight.

## Models & credits
- Voice clone + narration: **`openbmb/VoxCPM2`** (2B, tokenizer-free, 48kHz)
- Story generation: **`openbmb/MiniCPM5-1B`** (1B — Tiny Titan ≤4B)
- *(stretch)* Multilingual: `CohereLabs/tiny-aya-global` · Cover art: FLUX.2-klein · GPU: Modal
- All models ≤ 32B.

## Privacy
The parent's recording is processed in memory and **never stored on the server**. It is used only
to generate the narration for that session.

## Languages
English and **Hindi** are natively supported by VoxCPM2 and demoed here. Kannada is an
**experimental** mode fine-tuned on the parent's own voice — see `KANNADA_EXPERIMENT.md`.

## Links
- GitHub (Codex-built): `<ADD PUBLIC REPO URL>`
- Hugging Face Space: `<ADD SPACE URL>`
- Demo video: `<ADD VIDEO LINK>`

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Env vars (only for stretch features): `HF_TOKEN`, `COHERE_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
