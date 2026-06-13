# AGENTS.md — House Rules for Codex (READ FIRST, EVERY SESSION)

You are building **DreamVoice**, a bedtime-story app that clones a parent's voice and
narrates an AI-generated children's story in it. This file is binding. Follow it on every task.
The detailed work is in **`CODEX_TASKS.md`** — do tasks there in order.

---

## The 10 Rules

1. **One task at a time, in order.** Open `CODEX_TASKS.md`, find the first unchecked task,
   do *only* that task. Do not jump ahead or batch tasks.

2. **Commit after every task.** When a task's acceptance check passes:
   `git add -A && git commit -m "Tn: <short description>"`.
   Frequent, attributed commits are a graded requirement (OpenAI Codex track). Never let work sit uncommitted.

3. **Use ONLY these exact model IDs. Never invent or "correct" a model name:**
   - Voice clone + TTS: `openbmb/VoxCPM2` (smaller fallback: `openbmb/VoxCPM-0.5B`)
   - Story LLM: `openbmb/MiniCPM5-1B`
   - Multilingual LLM (stretch): `CohereLabs/tiny-aya-global`
   - ASR validation (stretch): `CohereLabs/cohere-transcribe-03-2026`
   - Cover art (stretch): FLUX.2-klein — use the exact ID from Black Forest Labs' current model card; if unsure, STOP.
   If any ID 404s or the load API differs from what you expect, **STOP** and write
   `# TODO(human): verify model API` — do not substitute a different model.

4. **Read the model card before coding against a model.** VoxCPM2's load/inference API
   must come from `huggingface.co/openbmb/VoxCPM2` (and its GitHub `OpenBMB/VoxCPM`), not from memory.
   The same for MiniCPM5-1B. Do not guess method names.

5. **Pin every dependency** in `requirements.txt` with `==`. Do **not** upgrade Gradio or torch
   mid-build. If a Gradio call errors, check the installed version's docs — don't guess new args.
   (Note: Gradio 6 uses `gr.Audio(sources=["microphone","upload"])` — `sources` is a list.)

6. **No secrets in code, ever.** Read keys from `os.environ` (`HF_TOKEN`, `COHERE_API_KEY`,
   `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`). Document required env vars in `README.md`.
   If a needed key is missing, STOP and ask the human — do not hardcode or fake it.

7. **Never persist the parent's raw voice recording server-side.** Work in a temp file,
   delete it in a `finally`. This is a privacy promise shown in the UI; keep it true.

8. **The app must launch at every commit.** If a feature is incomplete, hide it behind a
   feature flag (`ENABLE_FLUX`, `ENABLE_MODAL`, etc., default off) so `python app.py` still runs.

9. **Kid-safety in story generation.** The story system prompt must forbid violence, fear,
   death, and scary content; target age 6; end calm with a sleep cue. Never relax this.

10. **STOP-and-ask triggers** (write the question as a `# TODO(human):` comment and pause the task):
    missing API key · model 404 · model OOM/won't load · Gradio API mismatch · ambiguous
    requirement not covered here · anything that would require inventing a model/API name.

---

## After each task, print exactly one status line
`DONE Tn: <what you did> — verified by <how you checked it>`
or
`BLOCKED Tn: <why> — need: <what you need from human>`

---

## What "done" means for the whole project (north star)
On Chrome: record ~30s of a voice → choose a genre + mood → within a reasonable wait,
hear a ~90-second bedtime story **narrated in that cloned voice**, inside an animated
storybook UI, ending on a gentle goodnight. Deployed as a public Hugging Face Space,
with a public GitHub repo full of your (Codex) commits linked in the README.

## Project layout you will create
```
dreamvoice/
├── app.py            # Gradio 6 entrypoint (Phase 2)
├── tts.py            # VoxCPM2 clone + narrate (Phase 1)
├── story.py          # MiniCPM5-1B story generation (Phase 1)
├── audio_utils.py    # reference-clip validation/trim (Phase 1)
├── ui_html.py        # gr.HTML starfield/moon/storybook (Phase 2)
├── modal_app.py      # Modal GPU wrapper (Phase 3, optional)
├── requirements.txt
├── README.md
├── AGENTS.md         # this file
└── CODEX_TASKS.md    # your task list
```
Do not create files outside this layout without a task telling you to.
