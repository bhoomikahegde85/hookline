# Hugging Face Docker Space (GPU hardware). TRIBE v2 pins torch>=2.5.1,<2.7 and Python>=3.11; this base image has torch 2.6.
# Not yet built by Hookline: check the Space's build logs on first deploy.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface TRIBE_CACHE_DIR=/home/user/tribe_cache
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user handler.py app.py ./

EXPOSE 7860
CMD ["uvicorn", "app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
