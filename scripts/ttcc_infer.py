#!/usr/bin/env python3
"""TTCC retention inference runner. Runs in the TTCC environment only (see docs/ttcc.md), never in
.venv or .venv-models; Hookline calls it as a subprocess through ``hookline.ttcc``.

    python scripts/ttcc_infer.py probe --repo go_viral
    python scripts/ttcc_infer.py infer --request request.json --results results.jsonl

``probe`` prints one JSON object: library versions, CUDA devices, the go_viral commit and whether
inference can run here. ``infer`` loads the checkpoint once and predicts every job in the request,
writing one raw-output JSON per job and one line per job to ``--results`` as it goes, so a crash
midway keeps the finished videos.

The inference path reproduces the protocol behind the checkpoint's published evaluation
(``examples/train/grpo/qwen2_5_omni_ttcc/rl/srcc_eval.py`` in go_viral@ttcc-rl):
register.py plugin -> ``get_model_processor(ckpt, model_type='qwen2_5_omni_retention')`` ->
template ``qwen2_5_omni_retention`` in train mode -> V8 row (short prompts, ``videos=[mp4]``,
``audios=[]``, assistant ``<cot></cot>``) -> one forward pass -> ``out.r_pred`` (shape (1, 60)).
Nothing here substitutes another model: the checkpoint files are verified against pinned sizes and
hashes, and the retention-head weights in memory must equal the ones stored in the checkpoint.

This file must import without torch (Hookline's tests read its constants); every heavy import is
inside a function.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

RUNNER_VERSION = "1"
MODEL_TYPE = "qwen2_5_omni_retention"
TEMPLATE_TYPE = "qwen2_5_omni_retention"
PLUGIN_RELPATH = "examples/custom/qwen2_5_omni_retention/register.py"
T_MAX = 60  # head outputs R(1..60)
SMALL_FILE_LIMIT = 64 << 20  # files below this are always sha256-checked
CHECKPOINT_REQUIRED = (  # loader inputs (validate_ckpt.py lists processor_config.json too; the v8 repo has none)
    "config.json", "generation_config.json", "model.safetensors.index.json", "preprocessor_config.json",
    "spk_dict.pt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "added_tokens.json", "chat_template.json",
)


class RunnerError(RuntimeError):
    """Inference cannot proceed; the message says what to fix."""


# ---------------------------------------------------------------- helpers (no torch)

def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def git_state(repo: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": None if status is None else bool(status)}


def code_identity(repo: Path) -> dict:
    plugin = repo / PLUGIN_RELPATH
    return {"repo": str(repo), **git_state(repo), "plugin": PLUGIN_RELPATH,
            "plugin_sha256": sha256_file(plugin) if plugin.is_file() else None,
            "runner_version": RUNNER_VERSION, "runner_sha256": sha256_file(Path(__file__))}


def verify_checkpoint_files(ckpt: Path, expected: dict[str, list], full_hash: bool) -> dict:
    """Check required files exist and pinned files match size (always) and sha256 (small files
    always, shards only with ``full_hash``). Raises ``RunnerError`` on any mismatch."""
    missing = [f for f in CHECKPOINT_REQUIRED if not (ckpt / f).is_file()]
    missing += [f for f in expected if not (ckpt / f).is_file() and f not in missing]
    if missing:
        raise RunnerError(f"checkpoint directory {ckpt} is missing {missing}; it is not a complete copy of the pinned checkpoint")
    checked = {}
    for name, (size, digest) in expected.items():
        path = ckpt / name
        actual_size = path.stat().st_size
        if actual_size != size:
            raise RunnerError(f"{path}: {actual_size} bytes, expected {size}; wrong or partial checkpoint file")
        if size <= SMALL_FILE_LIMIT or full_hash:
            actual = sha256_file(path)
            if actual != digest:
                raise RunnerError(f"{path}: sha256 {actual} != pinned {digest}; this is not the pinned checkpoint")
            checked[name] = {"size": size, "sha256": actual, "hash_verified": True}
        else:
            checked[name] = {"size": size, "sha256": digest, "hash_verified": False}
    return checked


def resolve_checkpoint(spec: dict) -> Path:
    """Local directory of the pinned checkpoint. Downloads only when the request allows it."""
    if spec.get("local_dir"):
        path = Path(spec["local_dir"]).expanduser().resolve()
        if not path.is_dir():
            raise RunnerError(f"checkpoint directory not found: {path}")
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RunnerError(f"huggingface_hub is not installed in the TTCC environment ({exc})") from exc
    try:
        return Path(snapshot_download(spec["repo_id"], revision=spec["revision"],
                                      local_files_only=not spec.get("allow_download", False)))
    except Exception as exc:  # LocalEntryNotFoundError, network errors
        hint = "" if spec.get("allow_download") else " (not in the local Hugging Face cache; pass --download-models to fetch ~9.4 GB)"
        raise RunnerError(f"could not resolve {spec['repo_id']}@{spec['revision']}: {type(exc).__name__}: {exc}{hint}") from exc


def contains_subsequence(ids: list[int], sub: list[int]) -> bool:
    """True if ``sub`` occurs in ``ids`` (the head anchors on the last ``</cot>``, else the last token)."""
    k = len(sub)
    return k > 0 and any(ids[i:i + k] == sub for i in range(len(ids) - k + 1))


def shrink_horizon(T: int, length: int | None, max_length: int) -> int:
    """Next whole-second window to try after a row of ``length`` tokens overflowed ``max_length``
    (video tokens grow about linearly with seconds); at least one second shorter."""
    if length and length > max_length:
        return min(T - 1, int(T * max_length / length))
    return T - 1


def is_oom(exc: BaseException) -> bool:
    """CUDA / MPS out-of-memory, as raised by torch."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def next_window_after_oom(T: int, min_T: int = 5, factor: float = 0.7) -> int | None:
    """The next, shorter opening window (whole seconds) to try after running out of memory; None below ``min_T``.
    Attention memory grows with the square of the input length, so a 30 % shorter window needs about half the memory."""
    new = min(T - 1, int(T * factor))
    return new if new >= min_T else None


def build_row(job: dict, prompts: dict) -> dict:
    """One V8 inference row (schema of make_v8_from_hf.py) with the bypass assistant span of srcc_eval.py.
    ``video_start`` / ``video_end`` (seconds; a window of a long video) limit decoding to that part of the file; ms-swift
    passes the row's ``chat_template_kwargs`` to qwen_omni_utils.fetch_video, which reads video_start/video_end. The prompt's
    T is the window length, so the model treats the window as a standalone ad."""
    row = {
        "messages": [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user_template"].format(T=job["T"])},
            {"role": "assistant", "content": prompts["assistant"]},
        ],
        "videos": [job["video_path"]],
        "audios": [],
        "ad_id": job["video_id"],
        "T": job["T"],
    }
    if job.get("video_end") is not None:
        row["chat_template_kwargs"] = {"video_start": float(job.get("video_start") or 0.0), "video_end": float(job["video_end"])}
    return row


# ---------------------------------------------------------------- probe

def machine_name() -> str:
    """CPU / SoC name, e.g. "Apple M2" (identifies the MPS or CPU device)."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine()


def library_versions() -> dict:
    from importlib import metadata
    out = {}
    for dist in ("torch", "torchvision", "transformers", "accelerate", "peft", "safetensors", "huggingface_hub",
                 "qwen-omni-utils", "decord", "eva-decord", "torchcodec", "av", "flash_attn", "numpy", "ms_swift"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def probe(repo: Path) -> dict:
    problems: list[str] = []
    info: dict = {"runner_version": RUNNER_VERSION, "python": platform.python_version(), "executable": sys.executable,
                  "platform": platform.platform(terse=True)}
    if not (repo / PLUGIN_RELPATH).is_file():
        problems.append(f"go_viral checkout not found: {repo / PLUGIN_RELPATH} is missing")
    info["code"] = code_identity(repo) if repo.is_dir() else None
    info["libraries"] = library_versions()
    for name, dist in (("torch", "torch"), ("transformers", "transformers"), ("qwen_omni_utils", "qwen-omni-utils"),
                       ("safetensors", "safetensors"), ("huggingface_hub", "huggingface_hub")):
        if info["libraries"].get(dist) is None:
            problems.append(f"{name} is not installed in this environment")
    if not any(info["libraries"].get(k) for k in ("decord", "eva-decord", "torchcodec", "torchvision")):
        problems.append("no video reader backend (decord, torchcodec or torchvision) is installed")
    info["machine"] = machine_name()
    cuda: dict = {"available": False}
    if info["libraries"].get("torch"):
        try:
            import torch
            cuda["available"] = bool(torch.cuda.is_available())
            if cuda["available"]:
                cuda["devices"] = [{"name": torch.cuda.get_device_name(i),
                                    "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                                    "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                                   for i in range(torch.cuda.device_count())]
                cuda["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
            cuda["torch_cuda"] = torch.version.cuda
            cuda["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception as exc:
            problems.append(f"torch import failed: {type(exc).__name__}: {exc}")
    info["cuda"] = cuda
    if repo.is_dir() and not problems:
        sys.path.insert(0, str(repo))
        try:
            import swift
            info["swift_file"] = getattr(swift, "__file__", None)
            if not str(Path(swift.__file__).resolve()).startswith(str(repo.resolve())):
                problems.append(f"swift is imported from {swift.__file__}, not from {repo}; install go_viral with "
                                f"pip install -e {repo} or remove the other ms-swift")
        except Exception as exc:
            problems.append(f"import swift (go_viral) failed: {type(exc).__name__}: {exc}")
    if not problems:
        try:
            from qwen_omni_utils import vision_process
            info["video_reader"] = vision_process.get_video_reader_backend()
        except Exception as exc:
            problems.append(f"qwen_omni_utils has no usable video reader: {type(exc).__name__}: {exc}")
    info["problems"] = problems
    info["inference_possible"] = not problems
    return info


# ---------------------------------------------------------------- inference

def _load_plugin(repo: Path):
    spec = importlib.util.spec_from_file_location("retention_plugin", repo / PLUGIN_RELPATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify_head(model, ckpt: Path, head_type: str) -> dict:
    """The retention head in memory must hold exactly the checkpoint's retention_head.* tensors
    (register.py restores them with strict=False and silently skips when they are absent)."""
    import torch
    from safetensors import safe_open

    head = getattr(model, "retention_head", None)
    if head is None:
        raise RunnerError("retention_head is not attached to the loaded model; register.py was not applied")
    if getattr(head, "head_type", None) != head_type:
        raise RunnerError(f"retention head type is {getattr(head, 'head_type', None)!r}, expected {head_type!r}")
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"]
    stored = {}
    for name in ("retention_head.linear.weight", "retention_head.linear.bias"):
        if name not in index:
            raise RunnerError(f"{name} is not in the checkpoint; refusing to predict with an untrained head")
        with safe_open(str(ckpt / index[name]), framework="pt") as f:
            stored[name] = f.get_tensor(name)
    live = {"retention_head.linear.weight": head.linear.weight, "retention_head.linear.bias": head.linear.bias}
    for name, tensor in stored.items():
        if tuple(tensor.shape) != tuple(live[name].shape) or not torch.equal(tensor.float().cpu(), live[name].detach().float().cpu()):
            raise RunnerError(f"{name} in memory differs from the checkpoint; the trained head was not restored")
    return {"verified": True, "head_type": head.head_type, "t_max": head.t_max,
            "weight_shape": list(stored["retention_head.linear.weight"].shape),
            "stored_dtype": str(stored["retention_head.linear.weight"].dtype)}


def encode_fitting(job: dict, prompts: dict, template, inputs_cls, max_length_error, max_length: int,
                   min_T: int = 5) -> tuple[dict, dict, dict | None]:
    """Encode the job's row. If it exceeds ``max_length`` and the job allows it, retry with the
    longest opening window (whole seconds, prompt T = window) that fits. Returns (row, encoding,
    window info or None when the requested horizon was used)."""
    import re

    row = build_row(job, prompts)
    try:
        return row, template.encode(inputs_cls.from_dict(row)), None
    except max_length_error as exc:
        if not job.get("shrink_to_fit"):
            raise
        first_error = str(exc)
        m = re.search(r"row\((\d+)\)", first_error)
        length = int(m.group(1)) if m else None
    T, tried = job["T"], []
    while True:
        T = shrink_horizon(T, length, max_length)
        if T < min_T:
            raise RunnerError(f"no opening window of at least {min_T} s fits max_length={max_length} "
                              f"(tried {tried}); first error: {first_error[:200]}")
        tried.append(T)
        row = build_row(job | {"T": T, "video_end": float(job.get("video_start") or 0.0) + T}, prompts)
        try:
            enc = template.encode(inputs_cls.from_dict(row))
        except max_length_error as exc:
            m = re.search(r"row\((\d+)\)", str(exc))
            length = int(m.group(1)) if m else None
            continue
        return row, enc, {"requested_T": job["T"], "T": T, "video_end_s": float(job.get("video_start") or 0.0) + T, "tried": tried,
                          "reason": f"the {job['T']} s input exceeded max_length={max_length} tokens ({first_error[:160]})"}


def _input_device(model):
    import torch
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def infer(request: dict, results_path: Path) -> int:
    settings = request["settings"]
    for key, value in settings["env"].items():  # read by qwen_omni_utils / ms-swift / register.py at load time
        os.environ[key] = str(value)
    mps_fallback = settings["device_map"] == "mps"
    if mps_fallback:  # ops MPS lacks run on the CPU instead of failing; set before torch is imported
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    repo = Path(request["repo"]).resolve()
    sys.path.insert(0, str(repo))
    started = time.time()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results = results_path.open("a")

    def emit(job: dict, status: str, error: str | None = None, **extra) -> None:
        results.write(json.dumps({"video_id": job["video_id"], "status": status, "error": error, **extra}) + "\n")
        results.flush()

    jobs = request["jobs"]
    try:
        import torch
        import swift
        if not str(Path(swift.__file__).resolve()).startswith(str(repo)):
            raise RunnerError(f"swift is imported from {swift.__file__}, not from {repo}")
        ckpt = resolve_checkpoint(request["checkpoint"])
        files = verify_checkpoint_files(ckpt, request["checkpoint"]["expected_files"],
                                        request["checkpoint"].get("verify_weights_sha256", False))
        plugin = _load_plugin(repo)
        from swift.model import get_model_processor
        from swift.template import get_template
        from swift.template.base import MaxLengthError
        from swift.template.template_inputs import TemplateInputs

        dtype = getattr(torch, settings["torch_dtype"])
        load_t0 = time.time()
        model, processor = get_model_processor(str(ckpt), torch_dtype=dtype, attn_impl=settings["attn_impl"],
                                               model_kwargs={"device_map": settings["device_map"]}, model_type=MODEL_TYPE)
        model.eval()
        load_seconds = time.time() - load_t0
        head = _verify_head(model, ckpt, settings["env"]["RETENTION_HEAD_TYPE"])
        tk = processor.tokenizer
        missing_tokens = [t for t in ("image_token", "video_token", "audio_token") if not hasattr(tk, t)]
        if missing_tokens:
            raise RunnerError(f"tokenizer lacks {missing_tokens}: the checkpoint's processor files are incomplete "
                              "(upstream fix: copy the tokenizer files of Qwen/Qwen2.5-Omni-3B into the checkpoint dir)")
        close_ids = plugin._find_close_cot_token_ids(tk)
        template = get_template(processor, max_length=settings["max_length"], template_type=TEMPLATE_TYPE,
                                remove_unused_columns=False)
        template.set_mode("train")  # as srcc_eval.py: keeps the assistant span (and its </cot> anchor) in context
        device = _input_device(model)
    except Exception as exc:
        msg = str(exc) if isinstance(exc, RunnerError) else f"{type(exc).__name__}: {exc}"
        for job in jobs:
            emit(job, "failed", f"model setup failed: {msg}", stage="setup")
        traceback.print_exc(file=sys.stderr)
        results.close()
        return 1

    provenance = {
        "runner_version": RUNNER_VERSION, "code": code_identity(repo), "libraries": library_versions(),
        "python": platform.python_version(), "platform": platform.platform(terse=True),
        "checkpoint": {"repo_id": request["checkpoint"]["repo_id"], "revision": request["checkpoint"]["revision"],
                       "local_dir": str(ckpt), "files": files},
        "retention_head": head, "model_type": MODEL_TYPE, "template_type": TEMPLATE_TYPE, "template_mode": "train",
        "device": str(device), "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "machine": machine_name(), "mps_cpu_fallback": mps_fallback,
        "settings": settings, "model_load_seconds": round(load_seconds, 2),
    }
    failures = 0
    for job in jobs:
        t0 = time.time()
        try:
            attempt, oom_tried = job, []
            while True:
                row, enc, window = encode_fitting(attempt, request["prompts"], template, TemplateInputs, MaxLengthError,
                                                  settings["max_length"])
                if not enc:
                    raise RunnerError("template.encode returned an empty encoding")
                batch = template.data_collator([enc])
                n_tokens = int(batch["input_ids"].shape[-1])
                anchor_found = contains_subsequence(batch["input_ids"][0].tolist(), close_ids)
                grid = batch.get("video_grid_thw")
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                try:
                    with torch.no_grad():
                        out = model(**batch)
                    break
                except Exception as exc:
                    smaller = next_window_after_oom(row["T"]) if (is_oom(exc) and job.get("shrink_to_fit")) else None
                    if smaller is None:
                        raise
                    oom_tried.append({"T": row["T"], "input_tokens": n_tokens})
                    del batch
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    attempt = job | {"T": smaller, "video_end": float(job.get("video_start") or 0.0) + smaller}
            if oom_tried:  # the prediction covers a shorter opening window than requested; the client records it
                window = {"requested_T": job["T"], "T": row["T"], "video_end_s": float(job.get("video_start") or 0.0) + row["T"],
                          "tried": [t["T"] for t in oom_tried] + [row["T"]],
                          "reason": "GPU out of memory at " + ", ".join(f"{t['T']} s ({t['input_tokens']} tokens)" for t in oom_tried)
                                    + (f"; earlier: {window['reason']}" if window else "")}
            r_pred = getattr(out, "r_pred", None)
            if r_pred is None:
                r_pred = model._retention_h_holder.r_pred
            if r_pred is None:
                raise RunnerError("forward produced no r_pred")
            values = r_pred[0].float().cpu().tolist()
            raw = {
                "video_id": job["video_id"], "video_path": job["video_path"], "T": row["T"], "window": window,
                "r_pred": values, "r_pred_shape": list(r_pred.shape), "r_pred_dtype": str(r_pred.dtype),
                "r_pred_meaning": "head output R(1..60); R(0)=1 is a convention, not an output",
                "input_tokens": n_tokens, "video_grid_thw": grid.tolist() if grid is not None else None,
                "anchor": "</cot>" if anchor_found else "last_token", "close_cot_token_ids": close_ids,
                "row": row, "inference_seconds": round(time.time() - t0, 3), "provenance": provenance,
            }
            write_json(Path(job["raw_output"]), raw)
            emit(job, "ok", raw_output=job["raw_output"], inference_seconds=raw["inference_seconds"])
        except MaxLengthError as exc:
            failures += 1
            emit(job, "failed", f"input exceeds max_length={settings['max_length']} tokens: {str(exc)[:300]}", stage="encode")
        except Exception as exc:
            failures += 1
            emit(job, "failed", f"{type(exc).__name__}: {str(exc)[:500]}", stage="inference")
            traceback.print_exc(file=sys.stderr)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
    results.close()
    print(json.dumps({"jobs": len(jobs), "failed": failures, "total_seconds": round(time.time() - started, 1)}))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe", help="print environment capabilities as JSON")
    p.add_argument("--repo", type=Path, required=True, help="go_viral checkout (ttcc-rl)")
    i = sub.add_parser("infer", help="run the jobs in a request file")
    i.add_argument("--request", type=Path, required=True)
    i.add_argument("--results", type=Path, required=True, help="JSONL, one line per job (appended)")
    args = ap.parse_args(argv)
    if args.cmd == "probe":
        print(json.dumps(probe(args.repo.expanduser().resolve())))
        return 0
    return infer(json.loads(args.request.read_text()), args.results)


if __name__ == "__main__":
    sys.exit(main())
