from __future__ import annotations

import os
from pathlib import Path

from transformers import TrainerCallback


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


class LoraEpochCheckpointCallback(TrainerCallback):
    def __init__(self, *, every_n_epochs: float, output_dir: Path) -> None:
        if every_n_epochs <= 0:
            raise ValueError("save_lora_every_n_epochs must be positive")
        self.every_n_epochs = every_n_epochs
        self.output_dir = Path(output_dir)
        self._save_interval_steps: int | None = None

    def _resolve_interval(self, args, state) -> None:
        if self._save_interval_steps is not None or state.max_steps <= 0:
            return
        steps_per_epoch = state.max_steps / max(float(args.num_train_epochs), 1.0)
        self._save_interval_steps = max(1, int(round(steps_per_epoch * self.every_n_epochs)))

    def on_step_end(self, args, state, control, **kwargs):
        if not _is_main_process():
            return control
        self._resolve_interval(args, state)
        if self._save_interval_steps is None:
            return control
        if state.global_step <= 0 or state.global_step % self._save_interval_steps != 0:
            return control
        steps_per_epoch = state.max_steps / max(float(args.num_train_epochs), 1.0)
        epoch_value = state.global_step / steps_per_epoch
        save_dir = self.output_dir / f"lora-epoch-{epoch_value:.3f}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model = kwargs.get("model")
        if model is None:
            return control
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.save_pretrained(save_dir)
        return control
