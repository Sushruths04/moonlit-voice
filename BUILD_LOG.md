# DreamVoice — Build Log

A running record of each implementation step and how it was tested.
Local tests = syntax compile + `python app.py` build & HTTP-serve check (no GPU).
GPU tests (actual model audio) are run separately on Modal/OpenCode and noted as PENDING-GPU.

| # | Step | Files | Local test | Result |
|---|------|-------|-----------|--------|
| Prior | Story length/quality fix | story.py | compile | ✅ |
| Prior | Kannada path (IndicF5 + IndicTrans2) | indic_text.py, indic_tts.py, modal_app.py | compile | ✅ build / PENDING-GPU audio |
| Prior | Per-sentence naturalness | modal_app.py | compile | ✅ build / PENDING-GPU audio |
| Prior | Kannada UI + output polish | app.py, ui_html.py | build+serve :7875 | ✅ 50,761 bytes |

## This round (Stage-2 polish + energy + tiles)

| # | Step | Files | Local test | Result |
|---|------|-------|-----------|--------|
| 1 | Loading "tucking you in" overlay + **error recovery** (generator flow) | app.py | build+serve :7876, unit-test error path | ✅ returns to Stage 1 + inline error on failure (no stuck loading) |
| 2 | Storybook **open-on-reveal** animation (rotateY book-open, replays each show) | ui_html.py | served, compile | ✅ |
| 3 | Animated **now-playing waveform** (9 CSS bars) in Stage 2 | app.py | served contains `dv-wave` | ✅ |
| 4 | **Energy slider** (calm↔energetic) threaded story.py → tts.py/indic_tts.py → modal_app.py (style + pause + story tone) | story.py, tts.py, indic_tts.py, modal_app.py, app.py | compile all, served `dv-slider` | ✅ build / PENDING-GPU audio effect |
| 5 | **Emoji genre/mood tiles** (label,value tuples; values stay clean) + glow/scale on select | app.py | served `🦁 Animals`, `dv-tiles` | ✅ |

### Notes / still PENDING-GPU (run on Modal via OpenCode)
- Energy now changes 3 things: story wording (story.py), VoxCPM2/IndicF5 style tags, and
  inter-sentence pause length (`_pause_for`). The *audible* effect needs a Modal run to confirm.
- **Modal redeploy required** again — `synthesize_story` and `synthesize_kannada` now take an
  `energy` arg, and `synthesize_kannada`/`translate_en_kn` are new. Run `modal deploy modal_app.py`.
- Storybook flip-to-back ("The End") face still exists but isn't triggered; the open animation is the active effect.
