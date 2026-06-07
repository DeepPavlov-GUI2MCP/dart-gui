---
name: hf-upload
description: Maps local dart-gui and gui_distillation folders to Hugging Face dataset repos and uploads eval rollouts or synthetic data. Use when uploading to HF, publishing inference results, synthetic data, or explaining Hub layout.
---

# Hugging Face Upload

Canonical map of **local folders → HF dataset repos**. Repo names mirror the git workspace: `{git-repo}.{subtree}`.

| Git workspace | HF dataset | Local root to upload from | Hub path rule |
|---------------|------------|---------------------------|---------------|
| `dart-gui` | [`tony-pitchblack/dart-gui.GUI-Docker-Env`](https://huggingface.co/datasets/tony-pitchblack/dart-gui.GUI-Docker-Env) | `GUI-Docker-Env/results_*` | strip `GUI-Docker-Env/` → `results_<run>/...` at dataset root |
| `gui_distillation` | [`tony-pitchblack/gui_distillation.data.synthetic`](https://huggingface.co/datasets/tony-pitchblack/gui_distillation.data.synthetic) | `data/synthetic/<app>/` | strip `data/synthetic/` → `<app>/...` at dataset root |

**Legacy:** `tony-pitchblack/dart-gui-eval-rollouts` still resolves but is deprecated. Always use `dart-gui.GUI-Docker-Env`.

**Related (not sourced from these workspaces):** [`tony-pitchblack/ubuntu_osworld_file_cache`](https://huggingface.co/datasets/tony-pitchblack/ubuntu_osworld_file_cache) — OSWorld task fixture/cache files referenced by eval JSON configs. Do not upload dart-gui or gui_distillation trees there unless explicitly asked.

---

## dart-gui → `dart-gui.GUI-Docker-Env`

### Upload

Only **`GUI-Docker-Env/results_*`** rollout directories. Each run is one top-level folder on the Hub.

| Local (dart-gui root) | Hub |
|-----------------------|-----|
| `GUI-Docker-Env/results_thunderbird_all_20260510_025056/` | `results_thunderbird_all_20260510_025056/` |
| `GUI-Docker-Env/results_holo_gpt54_writer_traces_20260607_031517/` | `results_holo_gpt54_writer_traces_20260607_031517/` |
| `GUI-Docker-Env/results_coact_*` | `results_coact_*/` |

**Do not upload:** `GUI-Docker-Env/` code, configs, `evaluation_examples/`, submodule internals, or the whole repo.

**Typical `results_*` contents:** agent traces under `pyautogui/` or `coact/`, screenshots, `traj.jsonl`, optional `preflight_augmented/`, `preflight_reasoning_cache/`, manifest JSONL files.

### Commands

Run from **dart-gui repo root** with `source .venv/bin/activate`. Confirm auth first (`hf auth whoami`).

List candidates:

```bash
python GUI-Docker-Env/scripts/upload_eval_rollouts.py --list
```

Upload explicit folders (script maps `GUI-Docker-Env/results_*` → `results_*` on Hub):

```bash
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  GUI-Docker-Env/results_<name>
```

Glob selection:

```bash
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  --glob "GUI-Docker-Env/results_coact_*"
```

Dry-run:

```bash
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  --dry-run GUI-Docker-Env/results_<name>
```

Large folders (~1GB+, many files): resumable upload. Stage on the **same filesystem** (hardlinks; symlinks are not traversed):

```bash
mkdir -p .tmp_hf_staging
cp -al GUI-Docker-Env/results_<name> .tmp_hf_staging/
hf upload-large-folder tony-pitchblack/dart-gui.GUI-Docker-Env .tmp_hf_staging --repo-type dataset
rm -rf .tmp_hf_staging
```

**Notes:** Each folder uploads independently. Missing paths fail by default (`--skip-missing` only when useful). `--include` / `--exclude` pass through to the HF API.

---

## gui_distillation → `gui_distillation.data.synthetic`

### Upload

Only **`data/synthetic/<app>/`** trees (generated synthetic task data).

| Local (gui_distillation root) | Hub |
|-------------------------------|-----|
| `data/synthetic/libreoffice_writer/` | `libreoffice_writer/` |

**Typical layout under `libreoffice_writer/`:** `active_root_<id>/classification/`, `task_generation/<model>/`, `yield_classification/`, OSWorld task JSON under run timestamps.

**Excluded by default:** `**/.logs/**` (via uploader script).

**Do not upload unless explicitly asked:** `data/source/` (fixtures), `data/maps/` (exploration graphs), `data/reports/`, `data/tmp/`.

Future apps: add `data/synthetic/<new_app>/` locally → upload to `<new_app>/` on the same HF repo.

### Commands

Run from **gui_distillation repo root**. Use **dart-gui `.venv`** for `huggingface_hub` (gui_distillation venv may lack it).

Default upload (`libreoffice_writer`):

```bash
source /home/pitchblack/dart-gui/.venv/bin/activate
cd /home/pitchblack/gui_distillation
python scripts/upload_synthetic_to_hf.py --no-create
```

Custom source or Hub subpath:

```bash
python scripts/upload_synthetic_to_hf.py \
  --source data/synthetic/libreoffice_writer \
  --path-in-repo libreoffice_writer \
  --commit-message "Update libreoffice_writer synthetic data"
```

---

## Auth and permissions

- Account: `tony-pitchblack`; stored token is usually `dart-gui-W` (fine-grained).
- Verify: `hf auth whoami`
- Fine-grained tokens need **`repo.write`** per target dataset. Renaming/creating repos requires namespace manage rights (not available on scoped tokens).
- If upload returns **403**, extend token access to the target repo or create the empty dataset on the Hub first, then grant write.

---

## Path mapping cheat sheet

```
dart-gui/GUI-Docker-Env/results_foo/bar.png
  → dart-gui.GUI-Docker-Env/results_foo/bar.png

gui_distillation/data/synthetic/libreoffice_writer/active_root_231/...
  → gui_distillation.data.synthetic/libreoffice_writer/active_root_231/...
```

When in doubt: **drop the git-repo prefix segment that appears in the HF repo name** (`GUI-Docker-Env/` or `data/synthetic/`), keep the rest of the relative path unchanged.

---

## Reporting

After an upload, report:

- source local path
- target repo + Hub prefix
- file count or size if known
- commit URL(s)
- any missing or failed selections
- anything skipped (e.g. `.logs/`) or blocked (permissions)
