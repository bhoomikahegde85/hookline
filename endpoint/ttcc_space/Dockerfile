# Hugging Face Docker Space (GPU hardware, 24 GB or more) for the TTCC retention model.
# NOT yet built or run by Hookline: check the build logs on first deploy. Attention uses sdpa (flash-attn is not installed),
# which the runner records as a deviation from upstream's published evaluation.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# The upstream code the checkpoint needs (register.py), pinned to the commit Hookline's adapter was written against.
RUN git clone --quiet -b ttcc-rl https://github.com/cliangyu/go_viral /home/user/go_viral \
 && git -C /home/user/go_viral checkout --quiet 893da03bccee6a18d221e96c5de55c9b35a8e350
RUN pip install --no-cache-dir -e /home/user/go_viral
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user app.py ttcc_infer.py ./

EXPOSE 7860
CMD ["uvicorn", "app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
