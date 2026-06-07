# SFT Dataset Template

`sft_messages.jsonl` uses OpenAI-style `messages` records for TRL chat SFT.

Upload this format to a Hugging Face dataset repo, then set `train_sft.dataset.repo_id` in `training/configs/sft_example.yml` or a copied config.
