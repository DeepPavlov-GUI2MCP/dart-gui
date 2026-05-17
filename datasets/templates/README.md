# SFT Dataset Template

`sft_messages.jsonl` uses OpenAI-style `messages` records for Axolotl `type: chat_template` SFT.

Upload this format to a Hugging Face dataset repo, then set `train_qlora.dataset.repo_id` in `training/configs/qlora_example.yml` or a copied config.
