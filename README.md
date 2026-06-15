---
title: DreamVoice
emoji: 🌙
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "6.18.0"
app_file: app.py
pinned: true
license: mit
tags:
  - thousand-token-wood
  - openbmb
  - minicpm
  - voxcpm2
  - voice-cloning
  - bedtime-stories
  - indicf5
  - indictrans2
---

# 🌙 DreamVoice — Bedtime stories in Mom's own voice

> Mom is traveling for work. Her 6-year-old can't sleep. She opens DreamVoice, taps
> **🐉 Dragons + 😄 Funny**, and hears *her mom's voice* — cloned, warm, familiar — tell a
> story about a clumsy dragon who sneezes glitter. She's asleep in three minutes, smiling.

DreamVoice clones a parent's voice from a short recording and narrates a custom, AI-generated
bedtime story in it — so a child hears a loved one read to them, even when they're far away.

**Built for the Build Small Hackathon (Hugging Face × Gradio, June 2026) — Thousand Token Wood track.**

## Source
Public GitHub repo: https://github.com/Sushruths04/moonlit-voice

## How it works
1. A parent records 30–60s of their voice (or uploads a clip).
2. **`openbmb/VoxCPM2`** clones the voice.
3. The child picks a **genre** (animals, kingdom, space, dragons, ocean, forest) and **mood**
   (magical, funny, calming, dreamy), and optionally names the hero.
4. **`openbmb/MiniCPM5-1B`** writes a ~290-word, age-6, calming bedtime story.
5. VoxCPM2 narrates the story **in the parent's cloned voice**, ending on a gentle goodnight.
6. For **Kannada**: English story → **`ai4bharat/indictrans2-en-indic`** translation → **`ai4bharat/IndicF5`** narration in the parent's voice.

## Models & credits

### Core models
| Model | Purpose | Size | License |
|-------|---------|------|---------|
| `openbmb/VoxCPM2` | English voice cloning + narration | 2B | OpenBMB |
| `openbmb/MiniCPM5-1B` | Story generation | 1B | OpenBMB |
| `ai4bharat/IndicF5` | Kannada voice cloning + narration | 0.4B | MIT |
| `ai4bharat/indictrans2-en-indic-1B` | English → Kannada translation | 1B | MIT |

### Fine-tuned models

| Model | Purpose | Base | Training |
|-------|---------|------|----------|
| [`mitvho09/IndicF5-Kannada-Bedtime-v2`](https://huggingface.co/mitvho09/IndicF5-Kannada-Bedtime-v2) | Kannada bedtime narration | `ai4bharat/IndicF5` | Full fine-tune, 800 clips, 500 steps |

### All models ≤ 32B ✅

## Privacy
The parent's recording is processed in memory and **never stored on the server**. It is used only
to generate the narration for that session.

## Languages
- **English** — VoxCPM2
- **ಕನ್ನಡ Kannada** — IndicF5 (fine-tuned) + IndicTrans2 translation

## Infrastructure
- **GPU**: Hugging Face Spaces GPU Zero (T4 16GB)
- All models run locally on GPU — no external API calls

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Env vars: `HF_TOKEN` (for gated models: IndicTrans2, IndicF5)
