# DreamVoice — Master Checklist & Strategy (Kannada, naturalness, story quality, UI prize)

**Deadline: 2026-06-15. Today: 2026-06-13.** Prioritize ruthlessly. Each item is tagged
🔴 must-win · 🟠 high-value · 🟡 nice-to-have. Boxes are for OpenCode/you to tick.

---

## 0. The key strategic decision (READ FIRST)

**Kannada voice cloning already exists and works zero-shot — use `ai4bharat/IndicF5`.**
- Trained on 1417h across 11 Indian languages **including Kannada**; clones a voice from a
  reference clip; MIT license; 24 kHz output. API:
  ```python
  from transformers import AutoModel
  model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
  audio = model(target_text, ref_audio_path="ref.wav", ref_text="<transcript of ref.wav>")
  ```
- **Catch:** IndicF5 needs the *transcript of the reference clip* (`ref_text`). So for Kannada mode
  the parent reads a **known sentence we show them** → we know `ref_text` exactly. Reliable + verifiable.

**Final model architecture:**
| Need | Model | Why |
|---|---|---|
| English narration (mom's voice) | **VoxCPM2** (openbmb) | OpenBMB prize alignment |
| **Kannada narration (mom's voice)** | **IndicF5** (ai4bharat) | Works today, near-human, verifiable |
| English story text | **MiniCPM5-1B** (→ upgrade to **MiniCPM4.1-8B**) | OpenBMB; 8B gives longer, richer stories |
| **Kannada story text** | English story → **IndicTrans2** (`ai4bharat/indictrans2-en-indic`) | SOTA En→Kn; guarantees correct Kannada text |
| Fine-tune for the **badge** | LoRA-finetune **VoxCPM2** on bedtime-story speech | Earns badge, improves model we actually use, OpenBMB alignment |

> Do NOT try to teach VoxCPM2 Kannada from scratch — its own GitHub issue #202 reports "garbage
> output" for new-language LoRA. We get the badge by fine-tuning VoxCPM2 on English bedtime stories
> instead (speaker adaptation, not language adaptation).

---

## 1. 🔴 Story quality & length (the "stories too short / not meaningful" fix)
- [x] `story.py`: force length with `min_new_tokens`, `repetition_penalty`, and an
  **outline → expand → continue-until-target** loop (a 1B model won't hit 400 words in one shot).
- [ ] **Upgrade the story model to `openbmb/MiniCPM4.1-8B`** (still OpenBMB, ≤32B) for noticeably
  richer, longer, more coherent stories. Keep MiniCPM5-1B as the low-RAM fallback. Run story gen
  on **Modal GPU** (same as TTS) so the HF Space isn't doing 8B on CPU.
- [ ] Verify: 3-min setting yields a 350–450 word, coherent, kid-safe story with a real arc.

## 2. 🔴 Kannada end-to-end (headline emotional feature)
- [x] `indic_text.py`: `translate_to_kannada(en_text)` via IndicTrans2 (Modal func).
- [x] `indic_tts.py`: `narrate_kannada(ref_wav, ref_text, kn_text)` via IndicF5 (Modal func),
  **per-sentence synthesis** for natural prosody (see §3).
- [x] UI Kannada flow: when "ಕನ್ನಡ Kannada" is selected, show the parent a **fixed sentence to read**
  (so we know `ref_text`), then: English story → translate → narrate in Kannada in mom's voice.
- [x] Deploy the IndicF5 + IndicTrans2 Modal functions (`modal deploy modal_app.py`).
- [x] Verify: IndicTrans2 translation ✅ (35s), IndicF5 narration ✅ (48s, 1.3MB). Full pipeline PASS.

## 3. 🟠 Naturalness (kills the "monotone, continuous-agent" tone; energetic for kids)
Root cause: synthesizing one long block in a single pass → flat, machine cadence. Fixes (implemented
in `tts.py` / `indic_tts.py`):
- [x] **Sentence-by-sentence synthesis** with short natural silences between sentences, then concat.
  This alone removes most of the "robot reading a wall of text" feel.
- [x] **Generate-N-pick** (VoxCPM2 docs recommend 1–3 tries) — synthesize each sentence up to N
  times and keep the cleanest (longest non-degenerate) take.
- [x] **Child-energetic style tags** per mood — brighter, more dynamic delivery for funny/magical;
  soft/slow for calming/dreamy. Vary slightly per sentence so it isn't a constant tone.
- [x] Tune `cfg_value` (expressiveness) and `inference_timesteps` (quality) on Modal; pick the pair
  that sounds most alive without artifacts. Document the chosen values.
- [x] Energy parameter (0-100) threaded through story → TTS; affects prompt wordiness + mood tags + pause counts.
- [ ] Optional: insert a tiny pitch/energy lift on exclamations and dialogue lines.

## 4. 🔴 Fine-tune VoxCPM2 on bedtime-story speech → "Fine-tuned model" merit badge
Scaffold in `finetune/` (self-improving / failproof — see §6 properties):
- [x] Download LJSpeech dataset (13,100 clips, ~24h audio) → build JSONL manifests.
- [x] LoRA fine-tune VoxCPM2 on Modal GPU (A100-80GB) — 6 experiments completed.
- [x] Auto-eval each checkpoint → **best: exp6 (r32, lr=5e-5, half data, val loss 0.872)**.
- [x] Compare stock vs LoRA audio → files in `comparison/` directory.
- [ ] Publish the LoRA/adapter to HF account `mitvho09/VoxCPM2-bedtime-lora` → **claim badge**.
- [ ] If quality beats stock VoxCPM2 for kids' stories, swap it into the app; else keep stock VoxCPM2.

## 4b. 🟠 Fine-tune IndicF5 on Kannada → "Fine-tuned model" merit badge (compatibility-gated)
**Stock IndicF5 remains the production Kannada path. Fine-tune runs as separate badge experiment.**
Detailed plan in `KANNADA_FINETUNE_PLAN.md`. Steps:
- [x] **Phase 1: Compatibility spike** — PASSED. IndicF5 loads as INF5Model; CFM training
      forward works; 337M DiT params trainable (96.1%); checkpoint save/reload works.
- [x] **Phase 2: Short domain fine-tune** — 300 steps on Rasa Kannada (18 clips).
      Final loss: 0.4553. Checkpoints saved every 50 steps.
- [x] **Phase 2b: Comparison audio** — Stock vs fine-tuned generated in `comparison_kannada/`.
- [x] **Phase 3: Publish** — Published to `mitvho09/IndicF5-Kannada-Bedtime` on HuggingFace.
      → **Claim badge** + link in README.

## 5. 🔴 UI/UX — win the design prize (current UI is "okay"; make it award-grade)
Stage 1 (record/choose): keep the starfield+moon, but:
- [ ] Replace the plain `gr.Audio` with a **pulsing record ring + live waveform** affordance.
- [ ] Make genre/mood real **tiles with emoji art** (not just styled radios); selected = amber glow + scale.
- [ ] Add an **"energetic ↔ calming" delivery slider** (maps to the §3 style + pace) — visible, playful.
Stage 2 (output) — the part you asked to add to:
- [ ] **Animated audio waveform** that reacts while the story plays.
- [ ] Trigger the **storybook 3D flip** on reveal (currently the flip CSS exists but never fires).
- [ ] **"Tucking you in…" loading animation** (moon + drifting Z's) during generation.
- [ ] Story text typeset like a real book page; **replay**, **download**, **create another** as soft moon/star buttons.
- [ ] 🟡 FLUX.2-klein **cover illustration** on the book cover (also Black Forest Labs prize + shareable).
- [ ] Mobile-responsive check (judges open on phones).

## 6. Properties every script must have (failproof / self-improving)
- Idempotent + **resumable** (checkpoint to a Modal Volume; re-running continues, never restarts from 0).
- **Self-checking**: after each stage assert a real artifact exists (non-empty wav / non-empty manifest /
  improving eval loss) and fail loud with a clear message if not.
- **Auto-retry** transient failures (model load, download) with backoff; **STOP-and-report** on real errors.
- No secrets in code; read from env. Everything callable headless so **OpenCode can run + verify it**.

## 7. Suggested build order for the remaining ~2 days
1. Story length/quality (§1) + Kannada pipeline (§2) wired into the app — get the demo *working*.
2. Naturalness pass (§3) — make it *sound* good.
3. UI output polish (§5 Stage 2) — make it *look* award-grade.
4. VoxCPM2 fine-tuning (§4) running in parallel on Modal the whole time (badge).
5. FLUX cover art + final README/video (§5 + submission).

## Models & exact IDs (do not change)
```
English TTS     openbmb/VoxCPM2
Kannada TTS     ai4bharat/IndicF5            (call: model(text, ref_audio_path=, ref_text=))
Story LLM       openbmb/MiniCPM5-1B  →  openbmb/MiniCPM4.1-8B (richer)
En→Kn text      ai4bharat/indictrans2-en-indic
Kannada data    OpenSLR SLR79 (Kannada)      https://openslr.org/79/
Cover art       FLUX.2-klein (BFL card)
```

---

## 8. Final Report — Verified 2026-06-14

### Pipeline Verification Results

| Pipeline | Status | Latency | Notes |
|----------|--------|---------|-------|
| Story generation (MiniCPM5-1B) | ✅ PASS | ~14s | 356 words in test; variable 100-400 |
| VoxCPM2 English narration | ✅ PASS | ~156s (cold) / ~6s (warm) | 1.7MB output, good bedtime pace |
| IndicTrans2 English→Kannada | ✅ PASS | ~35s | Correct Kannada translation |
| IndicF5 Kannada narration | ✅ PASS | ~48s | 1.3MB output, per-sentence concat |
| Energy comparison (10 vs 90) | ✅ PASS | ~150s each | Files generated, audible difference confirmed |
| End-to-end English | ✅ PASS | ~170s | story + TTS in sequence |
| End-to-end Kannada | ✅ PASS | ~230s | story + translate + IndicF5 |

### Chosen Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| VoxCPM2 `cfg_value` | 2.0 | Good expressiveness without artifacts |
| VoxCPM2 `inference_timesteps` | 10 | Balance of quality vs speed |
| VoxCPM2 default speed | 0.9 | Bedtime pace (~140 wpm) |
| IndicF5 per-sentence concat | Enabled | Natural prosody, avoids flat cadence |
| Energy range | 0-100 | Maps to mood tags + pause counts + wordiness |
| Story model | MiniCPM5-1B | OpenBMB alignment; 8B optional via env var |

### Modal Image Versions (Critical — do not change)

| Image | torch | transformers | Other | Why |
|-------|-------|-------------|-------|-----|
| Main (voxcpm) | 2.11.0 | 5.12.0 | voxcpm 2.0.3, accelerate 1.14.0 | VoxCPM2 requires transformers 5.x |
| IndicTrans2 | 2.11.0 | **4.51.3** | huggingface_hub unpinned | IndicTransToolkit broken on transformers 5.x (`PreTrainedTokenizerBase` import) |
| IndicF5 | **2.4.1** | **4.46.3** | vocos 0.1.0, f5-tts | vocos mel filter bank crash with torch 2.5+ and transformers 5.x |

### What Broke & How We Fixed It

1. **IndicF5 `torchaudio.functional.melscale_fbanks` crash** — `vocos` library creates mel filter bank tensors that land on `meta` device with torch 2.5+ / transformers 5.x. Fixed by pinning `torch==2.4.1 + transformers==4.46.3`.
2. **IndicTransToolkit `PreTrainedTokenizerBase` import** — transformers 5.x moved/renamed tokenizer base classes. Fixed by pinning `transformers==4.51.3` for the indictrans image.
3. **Gated model access** — IndicTrans2 and IndicF5 require accepted HuggingFace agreements. Fixed by using `mitvho09` account token in Modal secret `dreamvoice-secrets`.
4. **VoxCPM2 training script missing from PyPI** — `scripts/train_voxcpm_finetune.py` not in the pip package. Fixed by adding `.git_clone("https://github.com/OpenBMB/VoxCPM.git", "/VoxCPM")` to the finetune image.
5. **Energy parameter not wired** — `synthesize_story` Modal function didn't receive energy. Fixed by threading energy through `story.py → tts.py/indic_tts.py → modal_app.py`.

### Remaining Work

1. **VoxCPM2 fine-tuning** — Publish LoRA adapter to HF for badge (§4)
2. **IndicF5 Kannada fine-tuning** — ✅ DONE. Published to `mitvho09/IndicF5-Kannada-Bedtime`
3. **UI polish** — animated waveform, storybook flip trigger, loading animation (§5)
4. **MiniCPM4.1-8B upgrade** — richer stories (§1)
5. **FLUX cover art** — book illustration (§5)
