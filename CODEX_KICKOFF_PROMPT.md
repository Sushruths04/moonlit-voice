# CODEX_KICKOFF_PROMPT — paste this into Codex

You are building **DreamVoice**, a Gradio app that clones a parent's voice and narrates an
AI-generated children's bedtime story in it. You are working in this repository.

## Before you write any code
1. Read `AGENTS.md` in full and treat all 10 rules as binding.
2. Read `CODEX_TASKS.md` — this is your work list. You will do it **in order, one task at a
   time**, exactly as written.
3. Confirm you can run git and that your commits are attributed to you (Codex). If not, STOP and
   tell me.

## How to work (repeat this loop for every task)
For each task in `CODEX_TASKS.md`, starting at the first unchecked box:
1. State which task you are on (e.g. "Starting T1.2").
2. If the task touches a model, **open that model's Hugging Face card / GitHub repo and follow
   its real load + inference API** — do not guess method names. Use only these exact IDs:
   `openbmb/VoxCPM2`, `openbmb/MiniCPM5-1B` (stretch: `CohereLabs/tiny-aya-global`,
   `CohereLabs/cohere-transcribe-03-2026`, FLUX.2-klein). If an ID 404s or the API differs from
   what you expect, STOP and leave `# TODO(human): verify model API`.
3. Implement only that task.
4. **Test it before moving on:**
   - Backend tasks (Phase 1): write/run a small test that actually calls the function and asserts
     it returns the right type and a real output (e.g. an existing, non-empty 48kHz `.wav` for TTS;
     a non-empty ≤~160-word string for story gen). The smoke test in T1.4 must run green.
   - UI tasks (Phase 2): confirm `python app.py` launches with no error and the page renders.
   - Print the result. If it fails, fix it before continuing — never mark a task done on a failure.
5. When the acceptance check passes: tick the box in `CODEX_TASKS.md`, then
   `git add -A && git commit -m "Tn: <short description>"`.
6. Print one status line: `DONE Tn: <what> — verified by <how>`.
7. Move to the next task.

## Models & Hugging Face
- Download/load models from Hugging Face Hub by their exact IDs above. Use `huggingface_hub`
  (and `HF_TOKEN` from `os.environ` if a gated model needs it — never hardcode it).
- Load each heavy model **once** as a module global (lazy init), not per request.
- Keep `requirements.txt` pinned; if you must add a dep for a model, pin it and say why.
- If a model is too large to run where you're testing, note it, keep the code correct per the
  model card, and flag it for the human to verify on a GPU (Modal). Do not silently swap models.

## Stop at the checkpoints
`CODEX_TASKS.md` has ⏸ CHECKPOINT lines after Phase 1, Phase 2, and Phase 3. When you reach one,
print `DONE Phase N`, summarize what you built and how you tested it, and **STOP and wait for human
review.** Do not continue past a checkpoint on your own.

## Hard rules (from AGENTS.md — do not break)
- One task at a time; commit after each.
- Never invent or substitute model IDs/APIs.
- No secrets in code — read from `os.environ`.
- Never store the parent's raw voice recording on the server (temp file, delete in `finally`).
- The story prompt must stay kid-safe (age 6, no violence/fear/death/scary content).
- The app must launch at every commit (hide unfinished features behind flags).
- Do NOT add Kannada in T3.4 — Kannada is a separate, flag-gated experiment (T3.5).

## Start now
Begin with the first unchecked task in `CODEX_TASKS.md` (Phase 0 / Phase 1). Go.
