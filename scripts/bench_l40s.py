#!/usr/bin/env python3
"""L40S performance + precision benchmark across precision variants, per task.

Variants per task:
  - eager fp32          (reference for precision diff)
  - compiled fp32       (= production: WavLM compiled_static / emotion optimized)
  - fp16 (autocast)
  - fp16 + compile
  - fp8 eager (torchao Float8 dynamic; compile path blocked on torch 2.10 -> eager only)

Reports throughput (win/s), peak VRAM, and max|Δ| vs fp32 on the task's output arrays.
Synthetic seed (content-independent for throughput; representative for a relative precision diff).
"""
from __future__ import annotations
import argparse, gc, json, time, traceback
from pathlib import Path
import numpy as np, torch, torch.nn as nn

from audio_classification_playground.acoustic_events.inference.artifacts import SAMPLE_RATE
from audio_classification_playground.acoustic_events.inference.audio import frame_audio
from audio_classification_playground.acoustic_events.inference.models import (
    AffectPredictor, DisfluencyPredictor, EmotionPredictor)
from audio_classification_playground.acoustic_events.inference.runners import (
    AFFECT_WINDOW_SEC, DISFLUENCY_WINDOW_SEC, EMOTION_WINDOW_SEC, DEFAULT_HOP_SEC)
from audio_classification_playground.acoustic_events.inference.emotion_runtime import OPTIMIZED_EMOTION_BATCH_SIZE
from audio_classification_playground.vox_profile.wavlm_inference import compile_wavlm_backbone

WAVLM_FP8_SUFFIXES = {"q_proj","k_proj","v_proj","out_proj","intermediate_dense","output_dense"}

def seed_audio(sec=520):
    sr=SAMPLE_RATE; n=sr*sec; rng=np.random.default_rng(0)
    x=rng.standard_normal(n).astype(np.float32)*0.05; t=np.arange(n)/sr
    for f in (130,220,440): x+=0.1*np.sin(2*np.pi*f*t).astype(np.float32)
    return np.clip(x,-1,1).astype(np.float32)

def _free():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def _fp8_quantize_wavlm(model):
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    tgt=lambda m,n: isinstance(m,nn.Linear) and n.split('.')[-1] in WAVLM_FP8_SUFFIXES
    nq=sum(1 for n,m in model.named_modules() if tgt(m,n))
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(), filter_fn=tgt)
    return nq

def _fp8_quantize_all_linear(model):
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    tgt=lambda m,n: isinstance(m,nn.Linear)
    nq=sum(1 for n,m in model.named_modules() if tgt(m,n))
    quantize_(model, Float8DynamicActivationFloat8WeightConfig(), filter_fn=tgt)
    return nq

# ---- WavLM (affect/disfluency) ----
def bench_wavlm(task, windows, bs):
    cls = AffectPredictor if task=="affect" else DisfluencyPredictor
    keys = ("arousal","valence","dominance") if task=="affect" else ("fluency_logits","disfluency_type_logits")
    variants = [
        ("eager_fp32",   dict(wavlm_autocast_dtype=None,  wavlm_compile=False), False),
        ("compiled_fp32",dict(wavlm_autocast_dtype=None,  wavlm_compile=True),  False),  # = production
        ("fp16",         dict(wavlm_autocast_dtype="fp16",wavlm_compile=False), False),
        ("fp16_compile", dict(wavlm_autocast_dtype="fp16",wavlm_compile=True),  False),
        ("fp8_eager",    dict(wavlm_autocast_dtype=None,  wavlm_compile=False), "wavlm"),
        ("fp8_compile",  dict(wavlm_autocast_dtype=None,  wavlm_compile=False), "wavlm_compile"),
    ]
    return _run_variants(task, variants, lambda kw: cls("wavlm", device="cuda", batch_size=bs, **kw),
                         windows, bs, keys)

# ---- emotion2vec ----
def bench_emotion(samples, bs):
    win=EMOTION_WINDOW_SEC
    def build(kw):
        return EmotionPredictor(device="cuda", batch_size=bs, **kw)
    variants = [
        ("eager_fp32", dict(autocast_dtype=None,  compile_model=False), False),
        ("optimized",  dict(autocast_dtype=None,  compile_model=True, allow_tf32=True), False),  # = production
        ("fp16",       dict(autocast_dtype="fp16",compile_model=False), False),
        ("fp8_eager",  dict(autocast_dtype=None,  compile_model=False), "all"),
    ]
    rows=[]; ref=None
    warm = samples[:int((EMOTION_WINDOW_SEC+63*DEFAULT_HOP_SEC)*SAMPLE_RATE)]
    for label, kw, fp8 in variants:
        try:
            _free(); p=build(kw); nq=0
            if fp8: nq=_fp8_quantize_all_linear(p._model.model)  # FunASR AutoModel -> .model is the torch nn.Module
            p.predict_audio(warm, sample_rate=SAMPLE_RATE, window_sec=win, hop_sec=DEFAULT_HOP_SEC)
            torch.cuda.synchronize(); t=time.perf_counter()
            sc,_=p.predict_audio(samples, sample_rate=SAMPLE_RATE, window_sec=win, hop_sec=DEFAULT_HOP_SEC)
            torch.cuda.synchronize(); el=time.perf_counter()-t
            sc=np.asarray(sc,np.float32);
            if ref is None: ref=sc
            d=float(np.abs(sc-ref).max()) if sc.shape==ref.shape else -1.0
            pk=torch.cuda.max_memory_reserved()/2**30
            rows.append(dict(variant=label, win_per_s=len(sc)/el, peak_gib=pk, max_abs_diff_vs_fp32=d,
                             nan=bool(np.isnan(sc).any()), n_quantized=nq))
            print(f"  emotion {label:14s} {len(sc)/el:7.1f} win/s  peak={pk:.1f}GiB  max|Δ|={d:.2e}  q={nq}", flush=True)
            del p; _free()
        except Exception as e:
            rows.append(dict(variant=label, error=f"{type(e).__name__}: {str(e)[:150]}"))
            print(f"  emotion {label:14s} ERR {type(e).__name__}: {str(e)[:130]}", flush=True); _free()
    return rows

def _run_variants(task, variants, build, windows, bs, keys):
    rows=[]; ref=None
    for label, kw, fp8 in variants:
        try:
            _free(); p=build(kw); nq=0
            # for fp8 we must NOT have compiled in __init__; build sets compile=False already
            if fp8 in ("wavlm","wavlm_compile"):
                nq=_fp8_quantize_wavlm(p._model)
                if fp8=="wavlm_compile":
                    compile_wavlm_backbone(p._model, mode="default", dynamic=False)  # compile AFTER fp8 quantize
            p(windows[:bs]); torch.cuda.synchronize()
            t=time.perf_counter(); out=p(windows); torch.cuda.synchronize(); el=time.perf_counter()-t
            out={k:np.asarray(out[k],np.float32) for k in keys}
            if ref is None: ref=out
            d=max(float(np.abs(out[k]-ref[k]).max()) for k in keys)
            pk=torch.cuda.max_memory_reserved()/2**30
            nan=any(bool(np.isnan(out[k]).any()) for k in keys)
            rows.append(dict(variant=label, win_per_s=len(windows)/el, peak_gib=pk,
                             max_abs_diff_vs_fp32=d, nan=nan, n_quantized=nq))
            print(f"  {task:10s} {label:14s} {len(windows)/el:7.1f} win/s  peak={pk:.1f}GiB  max|Δ|={d:.2e}  q={nq}", flush=True)
            del p; _free()
        except Exception as e:
            rows.append(dict(variant=label, error=f"{type(e).__name__}: {str(e)[:150]}"))
            print(f"  {task:10s} {label:14s} ERR {type(e).__name__}: {str(e)[:130]}", flush=True); _free()
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["affect","disfluency","emotion"])
    ap.add_argument("--bs-wavlm", type=int, default=256)
    ap.add_argument("--windows", type=int, default=2048)
    ap.add_argument("--json-out")
    args=ap.parse_args()
    samples=seed_audio(520)
    report={"device":torch.cuda.get_device_name(0),"torch":torch.__version__,
            "bs_wavlm":args.bs_wavlm,"windows":args.windows,"tasks":{}}
    for task in args.tasks:
        print(f"\n##### {task} #####", flush=True)
        if task in ("affect","disfluency"):
            win_sec=AFFECT_WINDOW_SEC if task=="affect" else DISFLUENCY_WINDOW_SEC
            W=frame_audio(samples, sample_rate=SAMPLE_RATE, window_sec=win_sec, hop_sec=DEFAULT_HOP_SEC)[:args.windows]
            report["tasks"][task]=bench_wavlm(task, W, args.bs_wavlm)
        elif task=="emotion":
            report["tasks"][task]=bench_emotion(samples, OPTIMIZED_EMOTION_BATCH_SIZE)
    print("\n"+json.dumps(report,indent=2), flush=True)
    if args.json_out: Path(args.json_out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("DONE_L40S", flush=True)

if __name__=="__main__":
    main()
