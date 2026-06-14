# CODEX_TASKS.md — DreamVoice Build Plan

Read `AGENTS.md` first. Do tasks **in order, one at a time, commit after each**.
Tick the box (`[x]`) when a task's acceptance check passes. Print the `DONE Tn:` status line.

Legend: 🟥 MVP (must ship) · 🟧 Stretch (only after MVP is deployed) · ⏸ CHECKPOINT (human reviews — pause)

---

## Phase 0 — Bootstrap 🟥

- [x] **T0.1** Initialize the repo. Create `.gitignore` (Python, `.venv/`, `__pycache__/`, `*.wav`,
  `.env`, `tmp/`), and a `README.md` stub. Acceptance: `git status` clean after first commit.
- [x] **T0.2** Create `requirements.txt` with **pinned** versions: `gradio`, `torch`,
  the VoxCPM2 runtime (`voxcpm` and/or `transformers` per the VoxCPM2 model card),
  `huggingface_hub`, `soundfile`, `librosa`, `numpy`.
  Acceptance: `pip install -r requirements.txt` succeeds in a fresh venv. If a VoxCPM2 dep
  name is unclear, STOP and read `github.com/OpenBMB/VoxCPM`.
- [x] **T0.3** Add a `README.md` line reserving the **public GitHub repo URL** and the
  **HF Space URL** (placeholders the human will fill). Commit.
> Human handles: creating the GitHub repo, the HF Space in `build-small-hackathon`, and confirming
> Codex commit attribution. Do not attempt to create remote accounts.

---

## Phase 1 — Backend functions (no UI yet) 🟥

- [x] **T1.1 `audio_utils.py`** — `prepare_reference(path) -> str`:
  load audio, convert to mono, resample to what VoxCPM2 expects, trim silence, enforce
  5–60 s, reject empty/too-quiet clips with a clear `ValueError` message.
  Acceptance: returns a cleaned temp wav for a good clip; raises a friendly error for a 1-s/silent clip.
- [x] **T1.2 `story.py`** — `generate_story(genre, mood, hero_name="", language="en") -> str`
  using `openbmb/MiniCPM5-1B`. Strict system prompt:
  *"Write a bedtime story for a 6-year-old. ≤150 words. Structure: a gentle hook, three short
  story beats, a calming resolution, and a final sleepy good-night line. Warm and simple. No
  violence, no fear, no death, nothing scary."* Inject genre/mood/hero_name/language.
  Lazy-load the model once (module global). Acceptance: for each of {animals, kingdom, space,
  dragons, ocean, forest} × {magical, funny, calming, dreamy} returns a coherent ≤~160-word story.
- [x] **T1.3 `tts.py`** — `clone_and_speak(ref_wav, text, speed=0.9) -> str` using `openbmb/VoxCPM2`:
  load model once; clone from `ref_wav`; synthesize `text` at a slow bedtime pace; return a 48kHz wav path.
  Read the exact API from the VoxCPM2 model card — do not guess. Delete intermediate temp files.
  Acceptance: given a sample voice + a sentence, produces an audible wav that resembles the reference voice.
- [x] **T1.4** Smoke test script `scripts/smoke_test.py` that runs T1.1→T1.2→T1.3 end-to-end on a
  sample clip and prints the output wav path. Acceptance: runs without error, produces a playable wav.
- [ ] ⏸ **CHECKPOINT A** — Print `DONE Phase 1`. **Pause. Human reviews backend before any UI.**

---

## Phase 2 — MVP UI 🟥

- [x] **T2.1 `ui_html.py`** — `gr.HTML` blocks (scoped CSS+JS, no external libs): animated
  starfield, glowing moon, and a CSS 3D storybook flip container. Colors: bg `#0a0a1a→#1a1a3e`,
  accent amber `#f5c842`, 24px radii, serif/handwritten title font (Lora/Pacifico via Google Fonts).
  Acceptance: renders the starfield + moon when injected into a Gradio Blocks page.
- [x] **T2.2 `app.py`** — Gradio 6 `gr.Blocks`, single page, **3 stages** with `gr.State`:
  - Stage 1 "Record Mom's Voice": `gr.Audio(sources=["microphone","upload"])`, quiet-room guidance, privacy note.
  - Stage 2 "What story tonight?": genre **tiles** (6) + mood buttons (4) + optional hero-name textbox.
  - Stage 3 "Your Story": storybook shows the story text + `gr.Audio` player + "Create another".
  Wire: clip → `audio_utils.prepare_reference` → `story.generate_story` → `tts.clone_and_speak`.
  Show a "tucking you in…" loading state during inference. Acceptance: `python app.py` launches.
- [x] **T2.3** End-to-end manual run on Chrome: record 30s → dragons + funny → get a story
  narrated in the cloned voice, displayed in the storybook. Acceptance: full path works locally.
- [ ] ⏸ **CHECKPOINT B** — Print `DONE Phase 2`. **Pause. Human reviews + does the first Space deploy.**

---

## Phase 3 — Stretch (ONLY after MVP is deployed; strict order; commit each) 🟧

- [ ] **T3.1 (S1) Goodnight ritual + lullaby.** After the story, synthesize
  *"Goodnight {name}, I love you."* in the cloned voice and append a soft hum/lullaby outro.
  Acceptance: the narration ends with the goodnight line in the cloned voice.
- [ ] **T3.2 (S2) FLUX.2-klein cover art.** Behind `ENABLE_FLUX`. Generate one cover image from
  the story's genre/title; show it on the storybook cover. STOP if the model ID/access is unclear.
  Acceptance: with the flag on, a cover image appears; with it off, app is unaffected.
- [ ] **T3.3 (S3) Modal GPU.** Behind `ENABLE_MODAL`. `modal_app.py` wraps VoxCPM2 inference on a
  Modal GPU; `tts.py` calls the Modal endpoint when enabled, else runs locally. Acceptance: with
  the flag on + Modal creds set, narration is generated via Modal; with it off, local path still works.
- [ ] **T3.4 (S4) Multilingual.** Add a language selector (English, Hindi, + Aya-supported Indic).
  Use `CohereLabs/tiny-aya-global` for non-English story generation. Optionally validate the
  reference clip with Cohere Transcribe. Acceptance: selecting Hindi yields a Hindi story narrated by VoxCPM2.
  **Do NOT add Kannada here** — Kannada is the separate experiment (see `KANNADA_EXPERIMENT.md`).
- [ ] **T3.5 (S5) Kannada (experimental).** Behind `ENABLE_KANNADA_BETA`, default OFF. Only wire this
  up **if** the human supplies a fine-tuned VoxCPM2 Kannada adapter path. Label it clearly "experimental
  (beta)" in the UI. Acceptance: flag off by default; when on with an adapter, Kannada narration plays.
- [ ] ⏸ **CHECKPOINT C** — Print `DONE Phase 3`. **Pause for human review before final submission.**

---

## Phase 4 — Submission polish 🟥

- [ ] **T4.1** Final `README.md`: one-line pitch; the traveling-mom story; **exact model credits +
  licenses** (`openbmb/VoxCPM2`, `openbmb/MiniCPM5-1B`, and any stretch models); feature list;
  **GitHub repo link**; privacy note; demo-video link; HF Space frontmatter tags for the track + sponsors.
- [ ] **T4.2** Confirm all models used are ≤32B (note Tiny Titan: MiniCPM5-1B ≤4B). Confirm the
  GitHub history shows your commits. Print a final `DONE Phase 4` checklist.

---

## Quick reference — exact IDs (do not change)
```
VoxCPM2          openbmb/VoxCPM2          (fallback openbmb/VoxCPM-0.5B)
Story LLM        openbmb/MiniCPM5-1B
Multilingual     CohereLabs/tiny-aya-global
ASR              CohereLabs/cohere-transcribe-03-2026
Cover art        FLUX.2-klein (use BFL's current card ID; STOP if unsure)
```
