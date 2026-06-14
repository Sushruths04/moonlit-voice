# DreamVoice — Work Summary (for agent review)

## What DreamVoice is
A Gradio web app that clones a parent's voice and narrates AI-generated bedtime stories in it. Built for the OpenAI Codex track hackathon.

## Architecture
```
app.py            → Gradio 6.18.0 entry-point, 3-stage UI
ui_html.py        → CSS starfield, moon, storybook animations
story.py          → Story generation via openbmb/MiniCPM5-1B
tts.py            → Voice cloning via openbmb/VoxCPM2 (Modal GPU only)
audio_utils.py    → Reference clip validation/trimming
modal_app.py      → Modal GPU deployment for VoxCPM2
scripts/smoke_test.py → End-to-end backend test
RECORDING_GUIDE.md → Script for users to read when recording
```

## Models Used
| Model | ID | Purpose | Runs on |
|-------|----|---------|---------|
| VoxCPM2 | `openbmb/VoxCPM2` | Voice cloning + TTS | Modal A10G GPU |
| MiniCPM5-1B | `openbmb/MiniCPM5-1B` | Story generation | Local CPU/GPU |

## What's built and working

### Phase 0 (Bootstrap) ✅
- Repo initialized, `.gitignore`, `requirements.txt` pinned, README stub

### Phase 1 (Backend) ✅
- **audio_utils.py**: `prepare_reference(path)` — loads audio, mono 16kHz, trims silence, enforces 5-60s, validates volume
- **story.py**: `generate_story(genre, mood, hero_name, language)` — kid-safe stories via MiniCPM5-1B, strict system prompt (no violence/fear/death), supports English + Hindi
- **tts.py**: `clone_and_speak(ref_wav, text, speed, mood)` — VoxCPM2 voice cloning via Modal GPU, mood-based style tags
- **modal_app.py**: Deployed to Modal with A10G GPU, `synthesize_story()` accepts mood for expressive narration
- **scripts/smoke_test.py**: End-to-end test

### Phase 2 (UI) ✅
- **ui_html.py**: Animated starfield (120 twinkling dots), glowing moon with pulse animation, CSS 3D storybook flip container
  - Palette: `#0a0a1a→#1a1a3e`, amber `#f5c842`, 24px radii
  - Fonts: Lora (serif) + Pacifico (handwritten) via Google Fonts
  - Scoped CSS via `head=`, JS via `js_on_load=`
- **app.py**: 3-stage Gradio flow:
  - Stage 1: Record voice + genre/mood/language selectors + hero name
  - Loading: Built-in Gradio indicator
  - Stage 2: Storybook display + audio player + "Create Another"
  - Full pipeline wired: audio → prepare_reference → generate_story → clone_and_speak
  - `.queue()` enabled for click handlers

### Recent improvements
- **Language selector**: English or Hindi — story LLM generates in the chosen language
- **Mood-based voice style**: VoxCPM2 gets expressive parenthetical tags per mood:
  - magical → "whimsical whisper, storytelling tone"
  - funny → "playful and animated, cheerful tone"
  - calming → "very slow and soothing, soft whisper"
  - dreamy → "slow and dreamy, breathy whisper, lullaby-like"
- **Modal-only TTS**: No local GPU fallback, all inference on Modal A10G

## How to run
```bash
cd /home/laptop/App_development/Bedtime_Voice/dreamvoice
source /tmp/dreamvoice-t02-venv/bin/activate
python app.py
# → http://127.0.0.1:7870
```

## Environment requirements
- `HF_TOKEN` — Hugging Face token (for model downloads)
- `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` — Modal credentials (for GPU inference)
- Both must be set in env before running

## What's NOT done yet (Phase 3+)
- T3.1: Goodnight ritual + lullaby outro
- T3.2: FLUX.2-klein cover art (behind `ENABLE_FLUX`)
- T3.3: Modal GPU flag (behind `ENABLE_MODAL`) — currently always Modal
- T3.4: Multilingual selector with more languages via Tiny Aya
- T3.5: Kannada experimental (behind `ENABLE_KANNADA_BETA`, needs fine-tuned adapter)
- T4.1: Final README with credits, links, demo video
- T4.2: Model size confirmation, git history check

## Known limitations
- Story length is ~150 words (MiniCPM5-1B output cap)
- VoxCPM2 supports 30 languages natively; Kannada is NOT one of them
- Voice style tags are best-effort; VoxCPM2 docs say "generating 1-3 times is recommended"
- No persistence of recordings (by design — privacy promise)
