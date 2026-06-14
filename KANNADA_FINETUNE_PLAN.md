# IndicF5 Kannada Fine-Tuning Plan — Compatibility-Gated

## Goal

Fine-tune IndicF5 on Kannada bedtime-story data to improve voice quality for the competition.
Publish the adapter to HuggingFace for the "Fine-tuned model" badge.

**Critical constraint**: Stock `ai4bharat/IndicF5` remains the production Kannada path.
Fine-tuning runs as a separate badge experiment on Modal. Only swap into the app if it
clearly wins A/B comparison against stock.

---

## Key Facts

- **IndicF5** is based on **F5-TTS** (SWivid/F5-TTS), which has built-in training/finetuning
  support via HuggingFace Accelerate
- **IndicF5 repo** (AI4Bharat/IndicF5) = inference-only wrapper; **training code lives in F5-TTS**
- IndicF5 already trained on Kannada from Rasa, LIMMITS, IndicVoices-R, IndicTTS — we're
  continuing from there
- **Model size**: 0.4B params (small enough for single A100-80GB)

---

## Approach: Compatibility-Gated Full Fine-Tuning

**First prove F5-TTS training can resume from IndicF5 weights with a 1-step smoke test.
Only then run short full fine-tunes on Modal A100-80GB. Stock IndicF5 remains the
production fallback.**

Why full fine-tuning over LoRA:
- F5-TTS training scripts are designed for full fine-tuning (not LoRA)
- IndicF5 is only 0.4B params — full fine-tuning is feasible on A100-80GB
- Competition judges may value full fine-tuning more than LoRA
- Simpler implementation — use existing F5-TTS training code directly

---

## Phase 1: Compatibility Spike (MANDATORY — 1-2 hours)

**Do NOT proceed to full training until this passes.**

### Step 1.1: Setup Modal Image

```
Image: torch 2.4.1 + transformers 4.46.3 + F5-TTS from source
GPU: A10G (cheap, fast for smoke test)
```

### Step 1.2: 1-Step Smoke Test

1. Clone F5-TTS repo: `git clone https://github.com/SWivid/F5-TTS.git`
2. Install F5-TTS: `pip install -e .` (editable mode)
3. Load `ai4bharat/IndicF5` weights into F5-TTS training code
4. Prepare 2-4 Kannada audio clips (any clean Kannada speech)
5. Run **1 training step** on the tiny dataset
6. Save checkpoint
7. Reload checkpoint
8. Synthesize one fixed Kannada bedtime line:
   `"ಮಕ್ಕಳೇ, ನಿದ್ರೆ ಮಾಡಿ. ಚಂದ್ರನು ನಿಮಗಾಗಿ ಕಾಯುತ್ತಿದ್ದಾನೆ."`

### Step 1.3: Pass/Fail Criteria

| Check | Pass | Fail |
|-------|------|------|
| Weights load into F5-TTS | No errors | Key mismatch, missing layers |
| 1 train step completes | Loss decreases | NaN, crash, OOM |
| Checkpoint saves | File exists, >100MB | Empty or corrupted |
| Checkpoint reloads | Model loads without error | Load failure |
| Audio synthesis | Non-empty WAV, audible | Garbled, silent, crash |

**If ANY check fails → STOP. Do not spend hours debugging. Report to user.**

---

## Phase 2: Short Domain Fine-Tune (IF PHASE 1 PASSES)

### Step 2.1: Curated Dataset (1-5h calm Kannada)

**Do NOT start with all 135h.** Fine-tuning on generic Rasa/LIMMITS will not automatically
create "bedtime" style.

Start with:
- Filter Rasa Kannada for `NEUTRAL` emotion clips only (~2-3h)
- Filter LIMMITS Kannada for calm/clear read speech (~1-2h)
- Manual spot-check: remove noisy, fast, or emotional clips
- Target: **1-5h of clean, calm, slow Kannada speech**

If no calm subset available, use the full Rasa Kannada (~55h) but filter programmatically
for duration >3s and <15s (bedtime story sentence length).

### Step 2.2: Training Config (Draft)

```yaml
# IndicF5 Kannada fine-tuning
base_model: ai4bharat/IndicF5
task: kannada_bedtime_story

# Training
learning_rate: 1e-5  # Lower LR for continued training
batch_size: 8
max_steps: 300       # Start short, extend only if improving
warmup_steps: 50
save_every: 50
eval_every: 50

# Data
train_data: /data/kannada_train.csv
val_data: /data/kannada_val.csv
sample_rate: 24000  # IndicF5 output rate
```

### Step 2.3: Auto-Eval

- Synthesize fixed Kannada bedtime sentence at each checkpoint
- Log audio to Modal Volume
- **Compare stock vs fine-tuned** at each step
- **Only continue beyond 300 steps if samples clearly improve**

### Step 2.4: Decision Gate

| Outcome | Action |
|---------|--------|
| Fine-tuned sounds clearly better | Continue to 500-1000 steps; use in app |
| Fine-tuned sounds same or worse | Stop at 300 steps; publish for badge; keep stock in app |
| Fine-tuned sounds garbled/broken | Abort; report to user; keep stock in app |

---

## Phase 3: Publish (AFTER TRAINING COMPLETES)

### Step 3.1: Upload to HuggingFace

- Upload fine-tuned IndicF5 checkpoint (not full model, just the diff)
- Link in README.md
- Claim "Fine-tuned model" badge

### Step 3.2: Demo Decision

| Fine-tune quality | App uses |
|-------------------|----------|
| Clearly better than stock | Fine-tuned IndicF5 |
| Same or worse | Stock IndicF5 (fine-tune published for badge only) |

---

## Datasets Available

| Dataset | Hours | Quality | Access | Notes |
|---------|-------|---------|--------|-------|
| **Rasa Kannada** | ~55h | Studio, expressive | HuggingFace (CC BY 4.0) | Used in original IndicF5 training |
| **LIMMITS/SYSPIN Kannada** | ~80h | Studio, read speech | HuggingFace (CC BY 4.0) | Used in original IndicF5 training |
| **IndicVoices-R Kannada** | ~355h | Natural conversational | HuggingFace (AI4Bharat) | Used in original IndicF5 training |
| **IndicTTS Kannada** | ~7h | Studio, small | Request-based (IIT Madras) | Supplementary |
| **OpenSLR SLR79** | ~15-20h | Crowdsourced | Direct download (CC BY-SA 4.0) | Multi-speaker |

**Start**: Curated 1-5h calm subset from Rasa + LIMMITS
**Scale to**: Full Rasa + LIMMITS (~135h) only after tiny run improves audio

---

## Compute Requirements

| Item | Phase 1 (Spike) | Phase 2 (Training) |
|------|-----------------|---------------------|
| GPU | A10G (cheap) | A100-80GB |
| Time | 30-60 min | 1-3 hrs (300 steps) |
| Cost | ~$1-2 | ~$10-20 |
| Checkpoint | ~100MB (test) | ~1.6GB (full model) |

---

## Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Compatibility spike (smoke test) | 1-2 hrs |
| 2 | Curated dataset prep | 30 min |
| 3 | Short fine-tune (300 steps) | 1-3 hrs |
| 4 | Eval + decision gate | 30 min |
| 5 | (Optional) Extend to 500-1000 steps | 2-4 hrs |
| 6 | Publish to HuggingFace | 15 min |
| **Total** | | **~4-10 hrs** |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| F5-TTS training code can't load IndicF5 weights | **BLOCKER** | Phase 1 smoke test catches this early |
| Full fine-tune degrades zero-shot cloning | Medium | Compare stock vs fine-tuned; keep stock as fallback |
| Generic data improves pronunciation but not bedtime prosody | Medium | Filter for calm/clear clips; manual spot-check |
| Dataset licenses block publishing derivative checkpoints | High | Check CC BY 4.0 terms; use only permissively licensed data |
| Training produces badge but not better demo | Low | Publish for badge; keep stock in app |
| OOM on A100-80GB | Low | Reduce batch size; use gradient accumulation |

---

## What This Earns

- **"Fine-tuned model" badge** for IndicF5 (Kannada) — if training completes
- **Better Kannada narration quality** — only if fine-tune wins A/B comparison
- **Both badges**: VoxCPM2 LoRA (English) + IndicF5 fine-tune (Kannada)

---

## Rules

1. **Stock IndicF5 is always the production Kannada path** unless fine-tune clearly wins
2. **Never claim fine-tuned quality beats stock unless verified by human listener**
3. **Publish adapter even if no improvement** — badge is the goal, not perfection
4. **Stop immediately if Phase 1 fails** — do not debug for hours
5. **Always link to stock IndicF5 as fallback** in README
