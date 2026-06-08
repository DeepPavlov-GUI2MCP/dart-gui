# ML Space GPU job image for UI-TARS LoRA SFT (pytorch2).
# Build from repo root: bash jobs/sft/run_uitars_lora/build.sh
#
# Base image: py3.11-torch2.9.0:0.0.42 (bundled torch/CUDA; no pip torch reinstall)
# Fallback: py3.11-torch2.7.0:0.0.42
FROM cr.ai.cloud.ru/aicloud-base-images/py3.11-torch2.9.0:0.0.42

USER user

ENV VIRTUAL_ENV=/opt/dart-sft-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

USER root
RUN mkdir -p "${VIRTUAL_ENV}" && chown -R user:user "${VIRTUAL_ENV}"
USER user

RUN python3 -m venv --system-site-packages "${VIRTUAL_ENV}"

COPY jobs/sft/run_uitars_lora/requirements.txt /tmp/requirements.txt
RUN "${VIRTUAL_ENV}/bin/python3" -m pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/jovyan
