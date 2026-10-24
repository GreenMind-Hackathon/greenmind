#!/usr/bin/env python3
"""
GreenMind: make an INT8 W8A8 copy of the official MANGO weights for faster PREFILL on A100.

Why: our Final runs are prefill-bound (~10K prompt tok/s at peak = A100's 16-bit limit).
AWQ W4A16 speeds up decode but still computes prefill in BF16. W8A8 runs matmuls on the
A100's INT8 tensor cores (2x the 16-bit peak), so prefill should get ~1.5-2x faster.

Method: data-free W8A8 (int8 per-channel weights, int8 dynamic per-token activations),
made with llm-compressor (open source) from the official BF16 weights, during the event.
No calibration data is used, so no dataset rules apply. Kept in BF16 (not quantized):
lm_head, the vision tower, the MTP head and the tiny Gated-DeltaNet gate projections.
The MTP head weights are copied from the official checkpoint so MTP speculation still works.

Runs on CPU (no GPU needed, ~15-30 min, needs ~40 GB RAM + ~30 GB disk).
Do NOT run while a grader submission is in progress (it competes for CPU/RAM).

Usage on the VM:
    python3 -m venv ~/qenv && source ~/qenv/bin/activate
    pip install llmcompressor
    python3 quantize_int8.py --out ~/models/mango-w8a8
"""
import argparse, json, os, shutil, sys, time

BASE = "CMKL/MANGO1.5-Qwen3.5-9B"
IGNORE = ["lm_head", "re:.*visual.*", "re:.*vision.*", "re:.*mtp.*",
          "re:.*in_proj_ba$", "re:.*in_proj_a$", "re:.*in_proj_b$"]


def load_model(path):
    import torch
    import transformers as T
    last = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        if not hasattr(T, cls):
            continue
        try:
            m = getattr(T, cls).from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cpu",
                                                trust_remote_code=False)
            print(f"loaded with {cls}: {type(m).__name__}")
            return m
        except Exception as e:  # try the next class
            last = e
            print(f"{cls} failed: {str(e)[:200]}")
    raise SystemExit(f"could not load {path}: {last}")


def graft_mtp(base_dir, out_dir):
    """transformers usually drops mtp.* weights on load; copy them back unquantized."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    bidx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    oidx_p = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(oidx_p):
        oidx = json.load(open(oidx_p))
    else:  # single-file output: build an index
        f = [x for x in os.listdir(out_dir) if x.endswith(".safetensors")][0]
        with safe_open(os.path.join(out_dir, f), "pt") as s:
            oidx = {"metadata": {}, "weight_map": {k: f for k in s.keys()}}
    mtp_keys = [k for k in bidx if ".mtp." in k or k.startswith("mtp.") or "mtp" in k.split(".")[:2]]
    missing = [k for k in mtp_keys if k not in oidx["weight_map"]]
    print(f"MTP tensors in base: {len(mtp_keys)}, missing in output: {len(missing)}")
    if not missing:
        return
    tensors = {}
    for k in missing:
        with safe_open(os.path.join(base_dir, bidx[k]), "pt") as s:
            tensors[k] = s.get_tensor(k)
    save_file(tensors, os.path.join(out_dir, "model-mtp.safetensors"), metadata={"format": "pt"})
    for k in missing:
        oidx["weight_map"][k] = "model-mtp.safetensors"
    json.dump(oidx, open(oidx_p, "w"), indent=2)
    print(f"grafted {len(missing)} MTP tensors (BF16)")


def fix_config(base_dir, out_dir):
    cfg_p = os.path.join(out_dir, "config.json")
    cfg = json.load(open(cfg_p))
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
    if qc is not None:
        ign = qc.setdefault("ignore", [])
        for pat in IGNORE:
            if pat not in ign:
                ign.append(pat)
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    # copy tokenizer / chat template / processor files the save may have skipped
    for f in os.listdir(base_dir):
        src, dst = os.path.join(base_dir, f), os.path.join(out_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") and f != "model.safetensors.index.json" \
                and not os.path.exists(dst):
            shutil.copy(src, dst)
            print("copied", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/models/mango-w8a8"))
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()
    t0 = time.time()
    from huggingface_hub import snapshot_download
    base_dir = snapshot_download(a.base)
    print("base at", base_dir)

    model = load_model(base_dir)
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = QuantizationModifier(targets="Linear", scheme="W8A8", ignore=IGNORE)
    oneshot(model=model, recipe=recipe)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, save_compressed=True)
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(base_dir).save_pretrained(a.out)
    except Exception:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(base_dir).save_pretrained(a.out)

    graft_mtp(base_dir, a.out)
    fix_config(base_dir, a.out)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {a.out}")
    print("serve with:  MODEL=/models/" + os.path.basename(a.out.rstrip('/')) + " MTP=2 ... ./serve2.sh")


if __name__ == "__main__":
    sys.exit(main())
