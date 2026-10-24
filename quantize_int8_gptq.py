#!/usr/bin/env python3
"""
GreenMind: INT8 W8A8 with GPTQ calibration (more accurate than the data-free version).

Why: our data-free W8A8 (quantize_int8.py) is fast, but ~10-15 Final requests per run
loop until the 8,192-token cap (math/Thai). GPTQ uses calibration data to choose the
int8 rounding that best preserves each layer's output, so the model should stay closer
to the official BF16 weights and loop less. Same INT8 kernels -> same prefill speed.

Rules: derived from the official BF16 weights, during the event, open-source tool
(llm-compressor), OPEN calibration data only (HuggingFaceH4/ultrachat_200k, generic chat;
no benchmark/test questions).

NEEDS THE GPU (~30-60 min). Stop the vLLM server first, never run during a grader run.

Usage on the VM:
    docker stop vllm
    source ~/qenv/bin/activate
    nohup python3 quantize_int8_gptq.py --out ~/models/mango-w8a8-gptq > quant_gptq.log 2>&1 &
    tail -f quant_gptq.log
"""
import argparse, json, os, shutil, sys, time

BASE = "CMKL/MANGO1.5-Qwen3.5-9B"
IGNORE = ["lm_head", "re:.*visual.*", "re:.*vision.*", "re:.*mtp.*",
          "re:.*in_proj_ba$", "re:.*in_proj_a$", "re:.*in_proj_b$"]
DATASET = "HuggingFaceH4/ultrachat_200k"


def load_model(path):
    import torch, transformers as T
    last = None
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        if not hasattr(T, cls):
            continue
        try:
            m = getattr(T, cls).from_pretrained(path, dtype=torch.bfloat16)
            print(f"loaded with {cls}: {type(m).__name__}", flush=True)
            return m
        except Exception as e:
            last = e
            print(f"{cls} failed: {str(e)[:200]}", flush=True)
    raise SystemExit(f"could not load {path}: {last}")


def calib_dataset(tok, n, max_len):
    from datasets import load_dataset
    ds = load_dataset(DATASET, split=f"train_sft[:{n}]").shuffle(seed=42)

    def prep(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, max_length=max_len, truncation=True, add_special_tokens=False)
    return ds.map(prep, remove_columns=ds.column_names)


def decoder_layer_name(model):
    for name, mod in model.named_modules():
        if name.endswith("layers.0"):
            return type(mod).__name__
    return None


def graft_mtp(base_dir, out_dir):
    from safetensors import safe_open
    from safetensors.torch import save_file
    bidx = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))["weight_map"]
    oidx_p = os.path.join(out_dir, "model.safetensors.index.json")
    if not os.path.exists(oidx_p):
        print("no output index; skip MTP graft check"); return
    oidx = json.load(open(oidx_p))
    mtp = [k for k in bidx if "mtp" in k.split(".")[:3]]
    missing = [k for k in mtp if k not in oidx["weight_map"]]
    print(f"MTP tensors in base: {len(mtp)}, missing in output: {len(missing)}", flush=True)
    if not missing:
        return
    t = {}
    for k in missing:
        with safe_open(os.path.join(base_dir, bidx[k]), "pt") as s:
            t[k] = s.get_tensor(k)
    save_file(t, os.path.join(out_dir, "model-mtp-graft.safetensors"), metadata={"format": "pt"})
    for k in missing:
        oidx["weight_map"][k] = "model-mtp-graft.safetensors"
    json.dump(oidx, open(oidx_p, "w"), indent=2)
    print(f"grafted {len(missing)} MTP tensors", flush=True)


def fix_config(base_dir, out_dir):
    p = os.path.join(out_dir, "config.json")
    cfg = json.load(open(p))
    qc = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
    if qc is not None:
        ign = qc.setdefault("ignore", [])
        ign += [x for x in IGNORE if x not in ign]
    json.dump(cfg, open(p, "w"), indent=2)
    for f in os.listdir(base_dir):
        src, dst = os.path.join(base_dir, f), os.path.join(out_dir, f)
        if os.path.isfile(src) and not f.endswith(".safetensors") and \
                f != "model.safetensors.index.json" and not os.path.exists(dst):
            shutil.copy(src, dst); print("copied", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/models/mango-w8a8-gptq"))
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--smoothquant", action="store_true", help="also apply SmoothQuant (riskier on hybrid models)")
    a = ap.parse_args()
    t0 = time.time()

    import torch
    assert torch.cuda.is_available(), "GPU not visible"
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU free memory: {free:.1f} GiB", flush=True)
    if free < 30:
        raise SystemExit("GPU busy: stop the vLLM server first (docker stop vllm)")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    base_dir = snapshot_download(BASE)
    tok = AutoTokenizer.from_pretrained(base_dir)
    model = load_model(base_dir)
    ds = calib_dataset(tok, a.samples, a.max_len)
    layer = decoder_layer_name(model)
    print("decoder layer class:", layer, flush=True)

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier
    recipe = []
    if a.smoothquant:
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        recipe.append(SmoothQuantModifier(smoothing_strength=0.8))
    recipe.append(GPTQModifier(targets="Linear", scheme="W8A8", ignore=IGNORE, dampening_frac=0.01))

    kw = dict(model=model, processor=tok, dataset=ds, recipe=recipe, max_seq_length=a.max_len,
              num_calibration_samples=a.samples)
    if layer:
        kw["sequential_targets"] = [layer]
    try:
        oneshot(**kw)
    except TypeError:
        kw.pop("sequential_targets", None)
        oneshot(**kw)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, save_compressed=True)
    tok.save_pretrained(a.out)
    graft_mtp(base_dir, a.out)
    fix_config(base_dir, a.out)
    with open(os.path.join(a.out, "QUANTIZATION.md"), "w") as f:
        f.write(f"W8A8 INT8 (GPTQ{' + SmoothQuant' if a.smoothquant else ''}) of {BASE}\n"
                f"calibration: {DATASET} train_sft first {a.samples} samples, max_len {a.max_len}\n"
                f"kept BF16: {IGNORE}\nmade with llm-compressor during GreenMind hackathon by Team 14\n")
    print(f"done in {(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
