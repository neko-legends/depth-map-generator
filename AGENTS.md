# AGENTS.md — DepthMapAIGenerator

Guidance for AI coding agents (and humans) working in this repo.

## ⚠️ Vendored files — do NOT edit here

These files are **copied (vendored)** from the canonical repo
[`neko-legends/NekoLegendsAI-Shared`](https://github.com/neko-legends/NekoLegendsAI-Shared).
Editing them in this repo is wrong — your change will be **overwritten** the next
time the module is re-vendored, and it desyncs this app from the other apps in
the suite.

- `src-tauri/src/neko_store.rs` — shared local-AI store resolver.

If you need to change one of these:

1. Edit the **canonical** copy in `NekoLegendsAI-Shared/rust/`.
2. Re-vendor into every app and verify:
   ```
   python NekoLegendsAI-Shared/scripts/neko_suite_doctor.py --fix
   ```
3. A pre-commit hook (installed via `NekoLegendsAI-Shared/scripts/install-hooks.sh`)
   blocks commits when a vendored copy has drifted from canonical.

## This app

DepthMap AI Generator is a Tauri 2 + React/TS desktop app with a Python worker.
It batch-generates depth maps (and WebP) from images and folders using MDA and
Depth Anything models, locally.

- Frontend: `src/` (React 18, TypeScript strict).
- Backend: `src-tauri/src/lib.rs` (Tauri commands + local Agent API on port 17334).
- Worker: `python/da3_worker.py` (creates/uses an app-managed virtual environment,
  preserving any legacy `python-env`/`models` folders).

## Validate before committing

```
export PATH="$PATH:$HOME/.cargo/bin"   # cargo may not be on PATH
cd src-tauri && cargo check
cd .. && npm run build                 # tsc strict + vite
```

After renaming the app or changing agent ports, run the suite doctor:
`python ../NekoLegendsAI-Shared/scripts/neko_suite_doctor.py`
