---
name: hf-upload
description: Maps local dart-gui and gui_distillation folders to Hugging Face dataset repos and uploads eval rollouts or synthetic data. Use when uploading to HF, publishing inference results, synthetic data, or explaining Hub layout.
---

# Hugging Face Upload

Canonical map of **local folders → HF dataset repos**. Repo names mirror the git workspace: `{git-repo}.{subtree}`.

| Git workspace | HF dataset | Local root to upload from | Hub path rule |
|---------------|------------|---------------------------|---------------|
| `dart-gui` | [`tony-pitchblack/dart-gui.GUI-Docker-Env.results`](https://huggingface.co/datasets/tony-pitchblack/dart-gui.GUI-Docker-Env.results) | `GUI-Docker-Env/results/results_*` | strip `GUI-Docker-Env/` → `results/results_<run>/...` |
| `gui_distillation` | [`tony-pitchblack/gui_distillation.data.synthetic`](https://huggingface.co/datasets/tony-pitchblack/gui_distillation.data.synthetic) | `data/synthetic/<app>/` | strip `data/synthetic/` → `<app>/...` at dataset root |

**Legacy:** `tony-pitchblack/dart-gui-eval-rollouts` and `tony-pitchblack/dart-gui.GUI-Docker-Env` are deprecated. Use `dart-gui.GUI-Docker-Env.results`.

**Related (not sourced from these workspaces):** [`tony-pitchblack/ubuntu_osworld_file_cache`](https://huggingface.co/datasets/tony-pitchblack/ubuntu_osworld_file_cache) — OSWorld task fixture/cache files referenced by eval JSON configs.

---

## dart-gui → `dart-gui.GUI-Docker-Env.results`

### Local layout

Eval runners write to `GUI-Docker-Env/results/results_*`. See `GUI-Docker-Env/results/README.md`.

Before Hub upload, zip screenshots:

```bash
source .venv/bin/activate
cd GUI-Docker-Env
python results/scripts/zip_images.py results/results_<name>
```

After Hub download, restore screenshots:

```bash
python results/scripts/unzip_images.py results/results_<name>
```

### Upload

Only **`GUI-Docker-Env/results/results_*`** rollout directories.

| Local (dart-gui root) | Hub |
|-----------------------|-----|
| `GUI-Docker-Env/results/results_thunderbird_all_20260510_025056/` | `results/results_thunderbird_all_20260510_025056/` |
| `GUI-Docker-Env/results/results_holo_gpt54_writer_traces_20260607_031517/` | `results/results_holo_gpt54_writer_traces_20260607_031517/` |

**Do not upload:** runner code, loose `.png`/`.jpg` (use `images.zip` instead), or the whole `GUI-Docker-Env/` tree.

List candidates:

```bash
python GUI-Docker-Env/scripts/upload_eval_rollouts.py --list
```

Upload:

```bash
python GUI-Docker-Env/scripts/upload_eval_rollouts.py \
  GUI-Docker-Env/results/results_<name>
```

Large folders (~1GB+): stage with hardlinks, then `hf upload-large-folder tony-pitchblack/dart-gui.GUI-Docker-Env.results .tmp_hf_staging --repo-type dataset`.

The uploader syncs `.gitattributes` (ignores loose images) and excludes `**/*.png`, `**/*.jpg`, `**/*.jpeg` by default.

---

## gui_distillation → `gui_distillation.data.synthetic`

Only **`data/synthetic/<app>/`** trees. Default upload:

```bash
source /home/pitchblack/dart-gui/.venv/bin/activate
cd /home/pitchblack/gui_distillation
python scripts/upload_synthetic_to_hf.py --no-create
```

---

## Path mapping cheat sheet

```
dart-gui/GUI-Docker-Env/results/results_foo/bar.txt
  → dart-gui.GUI-Docker-Env.results/results/results_foo/bar.txt
```

---

## Reporting

After an upload, report source path, target repo + Hub prefix, file count, commit URL(s), and anything skipped or blocked.
