#!/usr/bin/env python3
# =============================================================================
# MULTILABEL PII — FRAME-WISE CLASSIFICATION USING CACHED WavLM FEATURES
# OOM-SAFE VERSION: COMPACT KEYS + ON-THE-FLY LABELS/WINDOWS (LRU)
# -----------------------------------------------------------------------------
# • Cache once per 30s audio chunk: WavLM encoder outputs (fp16 on disk)
# • Enumerate encoder frames via COMPACT KEYS (stem_id, frame_idx)
# • Build per-stem labels on the fly (tiny LRU of stems)
# • Read features for [i-CTX, i+CTX] by stitching cached chunks (tiny LRU)
# • Train epoch + Eval are strictly time-capped; eval sample-capped too
# • Val/Test are re-sampled to match TRAIN class mix (frame-level)
# • Designed to avoid OOM (no giant in-memory per-frame list)
# =============================================================================

import os, math, json, random, copy, time, gc
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import OrderedDict, defaultdict, deque

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from transformers import WavLMModel, get_scheduler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_recall_fscore_support, multilabel_confusion_matrix, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================
# PATHS (cluster) — FULL DATA
# ==========================
AUDIO_DIR   = "/cluster/work/users/tarasan/full_data"
JSON_DIR    = "/cluster/projects/nn9851k/tarasan/thesis/data/full_data_json_filtered"
MODEL_DIR   = "/cluster/projects/nn9851k/tarasan/thesis/models/wavlm/wavlm-base-plus"

# Cache lives in SCRATCH (/cluster/work/users/tarasan)
CACHE_DIR    = "/cluster/work/users/tarasan/cache_wavlm_fp16_fulldata"

# Results under the dedicated folder you specified
RESULTS_BASE = "/cluster/projects/nn9851k/tarasan/thesis/code/multilabel_classification/fullscale/plots_logs_models/wavlm_frame_mutiheadattn_fulldata"

# ==========================
# CONFIG (task)
# ==========================
CLASSES = ["DATE_TIME", "LOCATION", "NRP", "PERSON", "NONIDENTIFIER"]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASSES)}
ID_COLS = [CLASS_TO_IDX[k] for k in CLASSES if k != "NONIDENTIFIER"]
NONID_COL = CLASS_TO_IDX["NONIDENTIFIER"]

SAMPLE_RATE   = 16000
EXCLUDE_STEMS = {"aaron_susanna"}  # exclude everywhere

def get_wavlm_enc_fps():
    stride_samples = 320
    fps = SAMPLE_RATE / float(stride_samples)  # ~50.0 fps
    return fps, stride_samples

FRAME_STRIDE  = 1
CONTEXT_SEC   = 0.5         # ± context around frame center (sec)
CHUNK_SEC     = 30.0        # cache chunk length (sec)

# ==========================
# OOM-SAFE SPEED / EFFICIENCY
# ==========================
# Cap frames per file in TRAIN (subsample per file)
MAX_FRAMES_PER_FILE_TRAIN = 15000
FRAME_STEP_TRAIN          = 1
FRAME_STEP_EVAL           = 1

# *** LRUs (keep tiny) ***
MAX_CHUNK_CACHE_ITEMS   = 24   # fp16 features per 30s chunk
MAX_LABEL_CACHE_STEMS   = 8    # per-stem frame label tensors

# DATA LOADER — make it gentle on RAM
NUM_WORKERS        = 1
PIN_MEMORY         = False
PERSISTENT_WORKERS = False
PREFETCH_FACTOR    = 1

# TRAIN LIMITERS
EPOCHS                = 10
BATCH_SIZE            = 4          # smaller batch to be extra safe
GRAD_ACCUM_STEPS      = 8          # effective batch = 32
LR                    = 2e-4
WEIGHT_DECAY          = 0.01
SEED                  = 42
EPOCH_TIME_LIMIT_MIN  = 300        # strict cap per epoch
# Optional global cap on number of TRAIN frames per epoch (after subsampling)
MAX_TRAIN_FRAMES_TOTAL = 1_000_000  # set None for unlimited

# Eval caps
EVAL_TIME_LIMIT_MIN   = 70
MAX_EVAL_SAMPLES      = 1_200_000

# *** NEW: hard cap for eval/test key-building to prevent stalls/OOM ***
EVAL_KEY_CAP          = 250_000

# Mixed precision / TF32
ALLOW_TF32 = True

# ==========================
# MULTI-HEAD ATTENTION POOLING
# ==========================
ATTN_HEADS    = 6  # must divide WavLM hidden size (base-plus=768 ok)

class MHAttentionClassifier(nn.Module):
    def __init__(self, in_dim, num_labels, num_heads=ATTN_HEADS):
        super().__init__()
        assert in_dim % num_heads == 0, f"in_dim {in_dim} not divisible by num_heads {num_heads}"
        self.h  = int(num_heads)
        self.dh = in_dim // self.h
        self.score = nn.Linear(in_dim, self.h)             # [B,T,H]
        self.value = nn.Linear(in_dim, in_dim, bias=False) # [B,T,D]
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, num_labels)
        )

    def forward(self, feats, mask=None):
        B, T, D = feats.shape
        scores = self.score(feats).transpose(1, 2)  # [B,H,T]
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)     # [B,H,T]
        values  = self.value(feats).view(B, T, self.h, self.dh).transpose(1, 2)  # [B,H,T,Dh]
        pooled_h = torch.sum(weights.unsqueeze(-1) * values, dim=2)              # [B,H,Dh]
        pooled   = pooled_h.reshape(B, self.h * self.dh)                          # [B,D]
        return self.classifier(pooled)                                            # [B,C]

# ==========================
# UTILS
# ==========================
def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def tee_print(s, fh=None):
    print(s, flush=True)
    if fh: fh.write(str(s) + "\n"); fh.flush()

def scan_audio_files(audio_root: str):
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    idx = {}
    for root, _, files in os.walk(audio_root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                stem = os.path.splitext(f)[0]
                if stem in EXCLUDE_STEMS:  # exclude early
                    continue
                idx[stem] = os.path.join(root, f)
    return idx

def read_json_any(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "words" in data:
        return [data]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict) and "words" in d]
    return []

def extract_words_targets(phrases):
    out = []
    for ph in phrases:
        for w in ph.get("words", []):
            s = float(w.get("start", 0.0))
            e = float(w.get("end", 0.0))
            labs = [str(x).upper() for x in (w.get("labels", []) or [])]
            out.append((s, e, labs))
    return out

def seconds_to_frames(t_sec, enc_fps, frame_stride=1):
    return int(math.floor(t_sec * (enc_fps / frame_stride)))

def load_waveform_mono_16k(path: str):
    wav, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    return wav.mean(0)  # [T]

# ==========================
# PER-FRAME TARGETS
# ==========================
def build_frame_targets(words_targets, audio_len_sec, enc_fps, frame_stride):
    C = len(CLASSES)
    Nf = int(math.ceil(audio_len_sec * enc_fps / frame_stride))
    y = torch.zeros(Nf, C, dtype=torch.float32)

    for s, e, labs in words_targets:
        if e <= 0 or e <= s:
            continue
        s_idx = seconds_to_frames(max(0.0, s), enc_fps, frame_stride)
        e_idx = int(math.ceil(e * (enc_fps / frame_stride)))
        s_idx = max(0, min(Nf, s_idx))
        e_idx = max(0, min(Nf, e_idx))
        if e_idx <= s_idx:
            continue
        for l in labs:
            l = l.upper()
            if l in CLASS_TO_IDX and l != "NONIDENTIFIER":
                y[s_idx:e_idx, CLASS_TO_IDX[l]] = 1.0

    any_id = (y[:, ID_COLS].sum(dim=1) > 0).to(torch.float32)
    y[:, NONID_COL] = 1.0 - any_id
    return y  # [Nf, C]

# ==========================
# CACHING — per 30s chunk (fp16) using WavLM encoder
# ==========================
@torch.no_grad()
def cache_chunks_for_stem_wavlm(stem: str, audio_path: str, wavlm: WavLMModel,
                                enc_fps: float, frame_stride: int) -> int:
    wav = load_waveform_mono_16k(audio_path)
    T = wav.shape[0]
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    device = next(wavlm.parameters()).device
    saved = 0

    n_chunks = max(1, (T + chunk_samples - 1) // chunk_samples)
    for ci in range(n_chunks):
        start_samp = ci * chunk_samples
        end_samp   = min(T, start_samp + chunk_samples)
        chunk_start_sec = start_samp / SAMPLE_RATE
        out_path = os.path.join(CACHE_DIR, f"{stem}__chunk{ci:04d}.pt")
        out_path = os.path.join(CACHE_DIR, f"{stem}__chunk{ci:04d}.pt")  # (fixed)
        if os.path.exists(out_path):
            continue

        seg = wav[start_samp:end_samp].to(torch.float32).to(device)
        seg = seg.unsqueeze(0)  # [1, S]

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            enc = wavlm(input_values=seg).last_hidden_state  # [1, Tenc, D]
        feats = enc.squeeze(0)  # [Tenc, D]

        if frame_stride > 1:
            feats = feats[::frame_stride, :]

        feats = feats.to(torch.float16).cpu()  # fp16 on disk

        f_start = seconds_to_frames(chunk_start_sec, enc_fps, frame_stride)
        torch.save({
            "feats": feats,
            "stem": stem,
            "chunk_idx": ci,
            "chunk_start_sec": float(chunk_start_sec),
            "f_start": int(f_start),
            "enc_fps": float(enc_fps),
            "frame_stride": int(frame_stride),
        }, out_path)
        saved += 1
    return saved

@torch.no_grad()
def cache_all_needed_chunks_wavlm(samples, model_dir: str, enc_fps: float, frame_stride: int, log_fh=None):
    ensure_dirs(CACHE_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wavlm = WavLMModel.from_pretrained(model_dir, local_files_only=True).to(device)
    wavlm.eval()
    if hasattr(wavlm.config, "use_cache"):
        wavlm.config.use_cache = False
    wavlm.gradient_checkpointing_enable()

    stems_seen = set()
    total_new = 0
    for s in tqdm(samples, desc="Caching (WavLM, per 30s chunk)"):
        stem = s["stem"]
        if stem in stems_seen: continue
        stems_seen.add(stem)
        newc = cache_chunks_for_stem_wavlm(stem, s["audio_path"], wavlm, enc_fps, frame_stride)
        if newc > 0:
            total_new += newc
            tee_print(f"  cached {newc} chunks for {stem}", log_fh)
    tee_print(f"Total new chunks cached: {total_new}", log_fh)

# ==========================
# CHUNK MAP + METADATA
# ==========================
def list_chunks_for_stem(stem: str) -> List[Dict[str,Any]]:
    files = sorted([str(p) for p in Path(CACHE_DIR).glob(f"{stem}__chunk*.pt")])
    chunks = []
    for fp in files:
        d = torch.load(fp, map_location="cpu")
        Tf = int(d["feats"].shape[0])
        f_start = int(d["f_start"])
        f_end   = f_start + Tf
        chunks.append({"file": fp, "f_start": f_start, "f_end": f_end})
    chunks.sort(key=lambda x: x["f_start"])
    return chunks

def safe_audio_dur_sec(audio_path: str, fallback_f_end: int, enc_fps: float, frame_stride: int):
    try:
        ai = torchaudio.info(audio_path)
        return ai.num_frames / ai.sample_rate
    except Exception:
        return fallback_f_end / (enc_fps / frame_stride)

# ==========================
# OOM-SAFE KEYS (no giant entries list) — with eval/test cap
# ==========================
def make_keys_for_split(samples_split, enc_fps, frame_stride, ctx_sec, frame_step,
                        is_train=False, per_file_cap=MAX_FRAMES_PER_FILE_TRAIN,
                        global_cap=None, seed=SEED, log_fh=None):
    """
    Returns:
      stems_meta: list of dicts per stem_id:
          {"stem": str, "audio_path": str, "words_targets": list, "chunks": list, "dur_sec": float}
      keys: list of (stem_id:int, frame_idx:int)  -- COMPACT
    """
    rng = random.Random(seed)
    stems_meta = []
    keys = []

    cap_this_split = None if is_train else EVAL_KEY_CAP
    built = 0

    for s in tqdm(samples_split, desc=f"Build keys ({'train' if is_train else 'eval'})"):
        if (not is_train) and cap_this_split is not None and built >= cap_this_split:
            tee_print(f"[cap] reached {cap_this_split} keys for eval/test — stop building more.", log_fh)
            break

        stem = s["stem"]
        chunks = list_chunks_for_stem(stem)
        if len(chunks) == 0:
            tee_print(f"[WARN] No cache for {stem}, skipping", log_fh); continue

        dur_sec = safe_audio_dur_sec(s["audio_path"], chunks[-1]["f_end"], enc_fps, frame_stride)
        y_frames_len = int(math.ceil(dur_sec * enc_fps / frame_stride))
        if y_frames_len <= 0:
            continue

        # indices with step
        frame_idxs = list(range(0, y_frames_len, max(1, int(frame_step))))

        # per-file cap (TRAIN)
        if is_train and per_file_cap is not None and len(frame_idxs) > per_file_cap:
            step = len(frame_idxs) / float(per_file_cap)
            frame_idxs = [frame_idxs[int(i*step)] for i in range(per_file_cap)]

        stem_id = len(stems_meta)
        stems_meta.append({
            "stem": stem,
            "audio_path": s["audio_path"],
            "words_targets": s["words_targets"],
            "chunks": chunks,
            "dur_sec": float(dur_sec),
        })

        for idx in frame_idxs:
            keys.append((stem_id, idx))
            built += 1
            if (not is_train) and cap_this_split is not None and built >= cap_this_split:
                break

        if (not is_train) and cap_this_split is not None and built >= cap_this_split:
            tee_print(f"[cap] reached {cap_this_split} keys after adding stem {stem}.", log_fh)
            break

    # global cap to bound training steps
    if global_cap is not None and is_train and len(keys) > global_cap:
        rng.shuffle(keys)
        keys = keys[:global_cap]

    rng.shuffle(keys)
    return stems_meta, keys

# ==========================
# DATASET with tiny LRUs
# ==========================
class FrameKeysDataset(Dataset):
    def __init__(self, stems_meta: List[Dict[str,Any]], keys: List[Tuple[int,int]],
                 enc_fps: float, frame_stride: int, ctx_sec: float):
        assert len(keys) > 0, "Empty key list."
        self.stems = stems_meta
        self.keys  = keys
        self.enc_fps = float(enc_fps)
        self.frame_stride = int(frame_stride)
        self.ctx_sec = float(ctx_sec)

        # infer D from any chunk file
        d0 = torch.load(self.stems[0]["chunks"][0]["file"], map_location="cpu")
        self.D = int(d0["feats"].shape[1])
        self.C = len(CLASSES)

        # LRU: feature chunks (fp16 tensors) and label tensors per stem
        self._chunk_cache = OrderedDict()  # fp -> tensor
        self._label_cache = OrderedDict()  # stem_id -> y_frames (fp16)

    def __len__(self): return len(self.keys)

    def _get_chunk_tensor(self, fp: str) -> torch.Tensor:
        if fp in self._chunk_cache:
            t = self._chunk_cache.pop(fp)
            self._chunk_cache[fp] = t
            return t
        d = torch.load(fp, map_location="cpu")
        x = d["feats"]  # fp16 [Tf,D]
        self._chunk_cache[fp] = x
        if len(self._chunk_cache) > MAX_CHUNK_CACHE_ITEMS:
            self._chunk_cache.popitem(last=False)
        return x

    def _get_labels_for_stem(self, stem_id: int) -> torch.Tensor:
        if stem_id in self._label_cache:
            y = self._label_cache.pop(stem_id)
            self._label_cache[stem_id] = y
            return y
        meta = self.stems[stem_id]
        dur  = meta["dur_sec"]
        y = build_frame_targets(meta["words_targets"], dur, self.enc_fps, self.frame_stride)  # [Nf,C] fp32
        y = y.to(torch.float16)  # store fp16 in LRU
        self._label_cache[stem_id] = y
        if len(self._label_cache) > MAX_LABEL_CACHE_STEMS:
            self._label_cache.popitem(last=False)
        return y

    def _spans_for_context(self, stem_id: int, frame_idx: int) -> List[Tuple[str,int,int]]:
        meta = self.stems[stem_id]
        chunks = meta["chunks"]
        t_center = (frame_idx * self.frame_stride) / self.enc_fps
        seg_start = max(0.0, t_center - self.ctx_sec)
        seg_end   = min(meta["dur_sec"], t_center + self.ctx_sec)
        f0 = seconds_to_frames(seg_start, self.enc_fps, self.frame_stride)
        f1 = int(math.ceil(seg_end * (self.enc_fps / self.frame_stride)))
        if f1 <= f0: f1 = f0 + 1
        spans = []
        for ch in chunks:
            if ch["f_end"] <= f0: continue
            if ch["f_start"] >= f1: break
            fs = max(0, f0 - ch["f_start"])
            fe = min(ch["f_end"], f1) - ch["f_start"]
            if fe > fs:
                spans.append((ch["file"], int(fs), int(fe)))
        return spans

    def __getitem__(self, i):
        stem_id, fidx = self.keys[i]
        # labels
        y_all = self._get_labels_for_stem(stem_id)      # [Nf,C] fp16
        if fidx >= y_all.shape[0]:
            fidx = y_all.shape[0]-1
        y = y_all[fidx]                                  # [C] fp16

        # features for context (stitch)
        spans = self._spans_for_context(stem_id, fidx)
        parts = []
        for (fp, fs, fe) in spans:
            x = self._get_chunk_tensor(fp)[fs:fe, :]     # fp16
            parts.append(x)
        x = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]  # [T,D] fp16
        return {"feats": x.to(torch.float16), "labels": y.to(torch.float16)}

def collate_cached(batch, normalize=True):
    feats = [b["feats"].to(torch.float32) for b in batch]
    labels= [b["labels"].to(torch.float32) for b in batch]
    B = len(feats)
    Tm = max(x.shape[0] for x in feats)
    D  = feats[0].shape[1]
    C  = labels[0].shape[0]

    X = torch.zeros(B, Tm, D, dtype=torch.float32)
    M = torch.zeros(B, Tm, dtype=torch.bool)
    for i, x in enumerate(feats):
        t = x.shape[0]
        x = x.nan_to_num(0.0, posinf=1e4, neginf=-1e4)
        if normalize and t>0:
            m = x.mean(dim=0, keepdim=True)
            s = x.std(dim=0, keepdim=True, unbiased=False)
            s = torch.where(s < 1e-5, torch.ones_like(s), s)
            x = (x - m) / s
        X[i, :t] = x
        M[i, :t] = True
    Y = torch.stack(labels, dim=0)
    return {"feats": X, "mask": M, "labels": Y}

# ==========================
# METRICS & PLOTS
# ==========================
def tune_thresholds_by_f1(Y_true, Y_prob, grid=None):
    C = Y_true.shape[1]
    if grid is None: grid = np.linspace(0.05, 0.95, 19)
    thr = np.zeros(C, dtype=np.float32)
    for c in range(C):
        y = Y_true[:, c].astype(int); p = Y_prob[:, c]
        best, best_t = -1.0, 0.5
        for t in grid:
            pred = (p >= t).astype(int)
            _, _, f1v, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            if f1v > best: best, best_t = f1v, t
        thr[c] = best_t
    return thr

def apply_thresholds(Y_prob, thresholds):
    return (Y_prob >= thresholds.reshape(1, -1)).astype(int)

def compute_global_metrics(Y_true, Y_prob, thresholds):
    out = {}
    try:
        ap_per_class = average_precision_score(Y_true, Y_prob, average=None)
        out["mAP_macro"] = float(np.nanmean(ap_per_class))
    except Exception:
        out["mAP_macro"] = float("nan")
    try:
        out["AUC_macro"] = float(roc_auc_score(Y_true, Y_prob, average="macro"))
    except Exception:
        out["AUC_macro"] = float("nan")
    try:
        out["AUC_micro"] = float(roc_auc_score(Y_true, Y_prob, average="micro"))
    except Exception:
        out["AUC_micro"] = float("nan")

    Y_pred = apply_thresholds(Y_prob, thresholds)
    out["F1_macro"] = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    out["F1_micro"] = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
    return out, Y_pred

def plot_loss_curve(losses, save_path, title="Train BCE loss"):
    plt.figure(figsize=(6,4))
    plt.plot(range(1, len(losses)+1), losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(True); plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max()/2.0 if cm.size>0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j]>thresh else "black", fontsize=8)
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

@torch.no_grad()
def collect_probs_and_labels(model, loader, device, time_deadline=None, sample_cap=None):
    model.eval()
    Ps, Ys = [], []
    n_collected = 0
    for i, b in enumerate(tqdm(loader, desc="Collect (eval)")):
        if time_deadline is not None and time.time() >= time_deadline:
            break
        if sample_cap is not None and n_collected >= sample_cap:
            break

        X = b["feats"].to(device, non_blocking=False)
        M = b["mask"].to(device, non_blocking=False)
        Y = b["labels"].to(device, non_blocking=False)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(X, M)  # [B,C]
        P = torch.sigmoid(logits).cpu().numpy()
        Yc = Y.cpu().numpy()

        if sample_cap is not None and n_collected + P.shape[0] > sample_cap:
            need = sample_cap - n_collected
            if need > 0:
                Ps.append(P[:need]); Ys.append(Yc[:need]); n_collected += need
            break
        else:
            Ps.append(P); Ys.append(Yc); n_collected += P.shape[0]

        del X, M, Y, logits
        if (i & 31) == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    C = len(CLASSES)
    if len(Ps)==0:
        return np.zeros((0, C), np.float32), np.zeros((0, C), np.float32)
    return np.concatenate(Ps, axis=0), np.concatenate(Ys, axis=0)

# ==========================
# Balance helpers (val/test match train proportions)
# ==========================
def entry_group_from_label_tensor(lbl: torch.Tensor) -> int:
    ids = (lbl[:NONID_COL] > 0).nonzero(as_tuple=False).view(-1)
    if ids.numel() > 0:
        return int(ids[0].item())  # first positive identifier
    return NONID_COL

def compute_group_proportions_from_keys(stems_meta, keys, enc_fps, frame_stride, sample_cap=200_000):
    # build a tiny label-cache for counting
    label_cache = OrderedDict()
    def get_labels(stem_id):
        if stem_id in label_cache:
            y = label_cache.pop(stem_id); label_cache[stem_id] = y; return y
        meta = stems_meta[stem_id]
        y = build_frame_targets(meta["words_targets"], meta["dur_sec"], enc_fps, frame_stride).to(torch.float16)
        label_cache[stem_id] = y
        if len(label_cache) > MAX_LABEL_CACHE_STEMS:
            label_cache.popitem(last=False)
        return y
    counts = defaultdict(int)
    total = 0
    for (stem_id, fidx) in keys[:min(len(keys), sample_cap)]:
        y_all = get_labels(stem_id)
        if fidx >= y_all.shape[0]:
            continue
        g = entry_group_from_label_tensor(y_all[fidx].to(torch.float32))
        counts[g] += 1
        total += 1
    if total == 0:
        return {i:0.0 for i in range(len(CLASSES))}
    return {i: counts.get(i,0)/float(total) for i in range(len(CLASSES))}

def make_balanced_key_indices(stems_meta, keys, enc_fps, frame_stride,
                              target_prop, cap_samples=None, seed=SEED):
    rng = random.Random(seed)
    # Build per-group pools of indices into keys
    # (we need labels; use small LRU)
    label_cache = OrderedDict()
    def get_labels(stem_id):
        if stem_id in label_cache:
            y = label_cache.pop(stem_id); label_cache[stem_id] = y; return y
        meta = stems_meta[stem_id]
        y = build_frame_targets(meta["words_targets"], meta["dur_sec"], enc_fps, frame_stride).to(torch.float16)
        label_cache[stem_id] = y
        if len(label_cache) > MAX_LABEL_CACHE_STEMS:
            label_cache.popitem(last=False)
        return y

    group_to_idxs = defaultdict(list)
    for idx, (stem_id, fidx) in enumerate(keys):
        y_all = get_labels(stem_id)
        if fidx >= y_all.shape[0]: 
            continue
        g = entry_group_from_label_tensor(y_all[fidx].to(torch.float32))
        group_to_idxs[g].append(idx)

    for g in group_to_idxs:
        rng.shuffle(group_to_idxs[g])

    N_total = len(keys) if cap_samples is None else min(len(keys), int(cap_samples))
    desired = {g: int(round(target_prop.get(g,0.0)*N_total)) for g in range(len(CLASSES))}
    drift = N_total - sum(desired.values())
    if drift != 0:
        residual = {g: (target_prop.get(g,0.0)*N_total - desired[g]) for g in desired}
        for g,_ in sorted(residual.items(), key=lambda kv: kv[1], reverse=(drift>0))[:abs(drift)]:
            desired[g] += 1 if drift>0 else -1
    leftover = 0
    for g in range(len(CLASSES)):
        avail = len(group_to_idxs.get(g, []))
        if desired[g] > avail:
            leftover += desired[g] - avail
            desired[g] = avail
    if leftover > 0:
        spare = []
        for g in range(len(CLASSES)):
            avail = len(group_to_idxs.get(g, []))
            if avail > desired[g]:
                spare.append((g, avail - desired[g]))
        i = 0
        while leftover > 0 and len(spare) > 0:
            g, room = spare[i % len(spare)]
            take = min(room, leftover)
            desired[g] += take
            leftover -= take
            i += 1

    selected = []
    for g in range(len(CLASSES)):
        pool = group_to_idxs.get(g, [])
        k = min(desired.get(g, 0), len(pool))
        selected.extend(pool[:k])

    rng.shuffle(selected)
    if cap_samples is not None and len(selected) > cap_samples:
        selected = selected[:cap_samples]
    return selected

# ==========================
# MAIN
# ==========================
def main():
    # Repro & math modes
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED); cudnn.benchmark = True
    if ALLOW_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    run_tag = datetime.now().strftime("wavlm_framewise_mhattn_cached_oomsafe_%Y%m%d_%H%M%S")
    RUN_DIR  = os.path.join(RESULTS_BASE, run_tag)
    PLOTS_DIR= RUN_DIR
    ensure_dirs(CACHE_DIR, RESULTS_BASE, RUN_DIR)

    log_fh = open(os.path.join(RUN_DIR, "log.txt"), "a", buffering=1)
    tee_print(f"===== Run {run_tag} (CACHED + OOM-SAFE) =====", log_fh)
    tee_print(f"AUDIO_DIR  = {AUDIO_DIR}", log_fh)
    tee_print(f"JSON_DIR   = {JSON_DIR}", log_fh)
    tee_print(f"MODEL_DIR  = {MODEL_DIR}", log_fh)
    tee_print(f"CACHE_DIR  = {CACHE_DIR}", log_fh)
    tee_print(f"RESULTS    = {RUN_DIR}", log_fh)
    tee_print(f"Exclude    = {sorted(EXCLUDE_STEMS)}", log_fh)

    enc_fps, enc_stride_samples = get_wavlm_enc_fps()
    tee_print(f"WavLM enc fps ~ {enc_fps:.4f} (stride samples ~{enc_stride_samples})", log_fh)
    tee_print(f"Context ±{CONTEXT_SEC:.2f}s | Cache chunk = {CHUNK_SEC:.1f}s", log_fh)
    tee_print(f"LRUs: chunks={MAX_CHUNK_CACHE_ITEMS}, label_stems={MAX_LABEL_CACHE_STEMS}", log_fh)
    tee_print(f"Dataloader: workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}, pin={PIN_MEMORY}, persistent={PERSISTENT_WORKERS}", log_fh)
    tee_print(f"Train cap per-file={MAX_FRAMES_PER_FILE_TRAIN}, global={MAX_TRAIN_FRAMES_TOTAL}", log_fh)
    tee_print(f"Time caps: epoch={EPOCH_TIME_LIMIT_MIN}m, eval={EVAL_TIME_LIMIT_MIN}m, eval_samples={MAX_EVAL_SAMPLES}", log_fh)

    # Discover dataset (match stems; exclude specified)
    audio_idx = scan_audio_files(AUDIO_DIR)
    json_files = sorted([str(p) for p in Path(JSON_DIR).rglob("*.json")])
    samples = []
    for jpath in json_files:
        stem = Path(jpath).stem
        if stem in EXCLUDE_STEMS:
            continue
        apath = audio_idx.get(stem)
        if apath is None: continue
        phrases = read_json_any(jpath)
        words_targets = extract_words_targets(phrases)
        if not words_targets: continue
        samples.append({"stem": stem, "audio_path": apath, "json_path": jpath, "words_targets": words_targets})
    assert len(samples) > 0, "No matched (json,audio) pairs found."

    # Split by file
    random.shuffle(samples)
    n = len(samples); n_tr = int(0.8*n); n_va = int(0.9*n)
    train_s = samples[:n_tr]
    val_s   = samples[n_tr:n_va]
    test_s  = samples[n_va:] if n_va<n else samples[n_tr:]

    # Cache features (if missing)
    cache_all_needed_chunks_wavlm(train_s + val_s + test_s, MODEL_DIR, enc_fps, FRAME_STRIDE, log_fh)

    # Build COMPACT KEYS (OOM-safe)
    tr_stems, tr_keys = make_keys_for_split(
        train_s, enc_fps, FRAME_STRIDE, CONTEXT_SEC, FRAME_STEP_TRAIN,
        is_train=True, per_file_cap=MAX_FRAMES_PER_FILE_TRAIN,
        global_cap=MAX_TRAIN_FRAMES_TOTAL, seed=SEED, log_fh=log_fh
    )
    va_stems, va_keys = make_keys_for_split(
        val_s, enc_fps, FRAME_STRIDE, CONTEXT_SEC, FRAME_STEP_EVAL,
        is_train=False, per_file_cap=None, global_cap=None, seed=SEED+1, log_fh=log_fh
    )
    te_stems, te_keys = make_keys_for_split(
        test_s, enc_fps, FRAME_STRIDE, CONTEXT_SEC, FRAME_STEP_EVAL,
        is_train=False, per_file_cap=None, global_cap=None, seed=SEED+2, log_fh=log_fh
    )
    tee_print(f"Keys: train/val/test = {len(tr_keys)}/{len(va_keys)}/{len(te_keys)}", log_fh)
    assert len(tr_keys)>0 and len(va_keys)>0 and len(te_keys)>0, "Empty keys for some split."

    # ===== Match VAL/TEST mix to TRAIN mix =====
    train_mix = compute_group_proportions_from_keys(tr_stems, tr_keys, enc_fps, FRAME_STRIDE, sample_cap=200_000)
    train_mix_named = {CLASSES[i]: float(train_mix.get(i,0.0)) for i in range(len(CLASSES))}
    tee_print(f"Train frame mix (proportions, sampled): {train_mix_named}", log_fh)

    va_bal_idx = make_balanced_key_indices(va_stems, va_keys, enc_fps, FRAME_STRIDE, train_mix,
                                           cap_samples=MAX_EVAL_SAMPLES, seed=SEED+3)
    te_bal_idx = make_balanced_key_indices(te_stems, te_keys, enc_fps, FRAME_STRIDE, train_mix,
                                           cap_samples=MAX_EVAL_SAMPLES, seed=SEED+4)

    def mix_from_bal_indices(stems, keys, idxs):
        # small label LRU
        label_cache = OrderedDict()
        def get_labels(stem_id):
            if stem_id in label_cache:
                y = label_cache.pop(stem_id); label_cache[stem_id] = y; return y
            meta = stems[stem_id]
            y = build_frame_targets(meta["words_targets"], meta["dur_sec"], enc_fps, FRAME_STRIDE).to(torch.float16)
            label_cache[stem_id] = y
            if len(label_cache) > MAX_LABEL_CACHE_STEMS:
                label_cache.popitem(last=False)
            return y
        counts = defaultdict(int)
        tot = 0
        for kidx in idxs:
            stem_id, fidx = keys[kidx]
            y_all = get_labels(stem_id)
            if fidx >= y_all.shape[0]: 
                continue
            g = entry_group_from_label_tensor(y_all[fidx].to(torch.float32))
            counts[g]+=1; tot+=1
        tot = max(1, tot)
        return {CLASSES[i]: counts.get(i,0)/tot for i in range(len(CLASSES))}
    tee_print(f"Val mix (balanced view): {mix_from_bal_indices(va_stems, va_keys, va_bal_idx)}", log_fh)
    tee_print(f"Test mix (balanced view): {mix_from_bal_indices(te_stems, te_keys, te_bal_idx)}", log_fh)

    # Datasets
    tr_ds = FrameKeysDataset(tr_stems, tr_keys, enc_fps, FRAME_STRIDE, CONTEXT_SEC)
    va_ds_full = FrameKeysDataset(va_stems, va_keys, enc_fps, FRAME_STRIDE, CONTEXT_SEC)
    te_ds_full = FrameKeysDataset(te_stems, te_keys, enc_fps, FRAME_STRIDE, CONTEXT_SEC)
    # Balanced subsets for val/test
    va_ds = Subset(va_ds_full, va_bal_idx)
    te_ds = Subset(te_ds_full, te_bal_idx)

    # Loaders
    loader_kw = dict(num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS)
    if NUM_WORKERS > 0:
        loader_kw["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_cached, **loader_kw)
    val_loader   = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_cached, **loader_kw)
    test_loader  = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_cached, **loader_kw)

    # Model
    D = tr_ds.D
    C = len(CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MHAttentionClassifier(in_dim=D, num_labels=C, num_heads=ATTN_HEADS).to(device)

    # Class imbalance (pos_weight) — estimate from a sample of train keys to avoid building a giant tensor
    pos_counts = torch.zeros(C, dtype=torch.float64)
    total_frames = 0
    # small label LRU
    label_cache = OrderedDict()
    def get_labels(stem_id):
        if stem_id in label_cache:
            y = label_cache.pop(stem_id); label_cache[stem_id] = y; return y
        meta = tr_stems[stem_id]
        y = build_frame_targets(meta["words_targets"], meta["dur_sec"], enc_fps, FRAME_STRIDE).to(torch.float16)
        label_cache[stem_id] = y
        if len(label_cache) > MAX_LABEL_CACHE_STEMS:
            label_cache.popitem(last=False)
        return y
    sample_for_weights = min(len(tr_keys), 200_000)
    for (stem_id, fidx) in tr_keys[:sample_for_weights]:
        y_all = get_labels(stem_id)
        if fidx < y_all.shape[0]:
            pos_counts += y_all[fidx].to(torch.float32).double()
            total_frames += 1
    neg_counts = max(1, total_frames) - pos_counts
    pos_weight = torch.where(pos_counts>0, neg_counts/torch.clamp(pos_counts, min=1.0), torch.ones_like(neg_counts))
    pos_weight = pos_weight.to(device).float()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = max(1, math.ceil(len(train_loader)/GRAD_ACCUM_STEPS))
    scheduler = get_scheduler("linear", optimizer,
                              num_warmup_steps=max(100, steps_per_epoch),
                              num_training_steps=EPOCHS*steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Sanity
    b0 = next(iter(train_loader))
    with torch.no_grad():
        L0 = model(b0["feats"].to(device, non_blocking=False), b0["mask"].to(device, non_blocking=False))
    tee_print(f"Sanity — batch={b0['labels'].shape[0]}  logits={tuple(L0.shape)} (expect [B,{C}])", log_fh)

    # Train (save best)
    best_val_macro = -1.0
    best_state = None
    best_thr_vec = None
    epoch_losses = []

    for ep in range(1, EPOCHS+1):
        model.train()
        run = 0.0
        optimizer.zero_grad(set_to_none=True)
        ep_deadline = time.time() + EPOCH_TIME_LIMIT_MIN * 60.0

        pbar = tqdm(train_loader, desc=f"Epoch {ep}")
        for step, batch in enumerate(pbar, start=1):
            if time.time() >= ep_deadline:
                tee_print(f"[Epoch {ep}] Time cap {EPOCH_TIME_LIMIT_MIN} min reached; stopping epoch early.", log_fh)
                pbar.close(); break

            X = batch["feats"].to(device, non_blocking=False)
            M = batch["mask"].to(device, non_blocking=False)
            Y = batch["labels"].to(device, non_blocking=False)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(X, M)     # [B,C]
                loss = bce(logits, Y) / GRAD_ACCUM_STEPS

            if not torch.isfinite(loss):
                tee_print("[warn] Non-finite loss; skipping batch", log_fh)
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True); scheduler.step()
            run += loss.item() * GRAD_ACCUM_STEPS
            pbar.set_postfix({"loss": f"{(run/max(1,step)):.4f}"})

            del X, M, Y, logits, loss
            if (step & 63) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg = run / max(1, len(train_loader))
        epoch_losses.append(avg)
        tee_print(f"Epoch {ep} | Train BCE loss: {avg:.4f}", log_fh)

        # ---- Validation (capped, balanced) ----
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        val_deadline = time.time() + EVAL_TIME_LIMIT_MIN * 60.0
        P_val, Y_val = collect_probs_and_labels(
            model, val_loader, device,
            time_deadline=val_deadline,
            sample_cap=MAX_EVAL_SAMPLES
        )
        if Y_val.shape[0] > 0:
            thr_vec = tune_thresholds_by_f1(Y_val, P_val)
            f1_macro = f1_score(Y_val, (P_val >= thr_vec.reshape(1,-1)).astype(int),
                                average="macro", zero_division=0)
            tee_print(f"Val macro-F1 (tuned thr) [capped]: {f1_macro:.4f}  (N={Y_val.shape[0]})", log_fh)
            if f1_macro > best_val_macro:
                best_val_macro = f1_macro
                best_state = copy.deepcopy(model.state_dict())
                best_thr_vec = thr_vec.copy()
                tee_print("Updated BEST head.", log_fh)

    # Plot loss
    loss_png = os.path.join(PLOTS_DIR, "loss_curve.png")
    plot_loss_curve(epoch_losses, loss_png, "Train BCE loss (frame-wise, WavLM MH-Attn, cached OOM-safe)")
    tee_print(f"Saved loss curve: {loss_png}", log_fh)

    # Save BEST model (+ thresholds)
    assert best_state is not None and best_thr_vec is not None, "No best model captured."
    ckpt_path = os.path.join(RUN_DIR, "model_best.pth")

    torch.save({
        "state_dict": best_state,
        "meta": {
            "in_dim": int(D),
            "num_labels": C,
            "num_heads": ATTN_HEADS,
            "enc_fps": float(enc_fps),
            "frame_stride": int(FRAME_STRIDE),
            "context_sec": float(CONTEXT_SEC),
            "class_names": CLASSES,
            "encoder": "WavLM",
            "model_dir": MODEL_DIR,
        },
        "best_thresholds": {CLASSES[i]: float(best_thr_vec[i]) for i in range(C)}
    }, ckpt_path)
    tee_print(f"Saved best model: {ckpt_path}", log_fh)

    # Load best for final eval
    model.load_state_dict(best_state)

    # ---- Final VAL (balanced view) ----
    val_deadline = time.time() + EVAL_TIME_LIMIT_MIN * 60.0
    P_val, Y_val = collect_probs_and_labels(
        model, val_loader, device,
        time_deadline=val_deadline,
        sample_cap=MAX_EVAL_SAMPLES
    )
    if Y_val.shape[0] > 0:
        metrics_val, _ = compute_global_metrics(Y_val, P_val, best_thr_vec)
        with open(os.path.join(RUN_DIR, "metrics_val.json"), "w") as f:
            json.dump({"classes": CLASSES, "val": metrics_val, "N": int(Y_val.shape[0]),
                       "val_mix_matched_train": True,
                       "train_mix": {CLASSES[i]: float(train_mix.get(i,0.0)) for i in range(len(CLASSES))}},
                      f, indent=2)

    # ---- Final TEST (capped, balanced view) ----
    test_deadline = time.time() + EVAL_TIME_LIMIT_MIN * 60.0
    P_test, Y_test = collect_probs_and_labels(
        model, test_loader, device,
        time_deadline=test_deadline,
        sample_cap=MAX_EVAL_SAMPLES
    )
    metrics, Yhat_test_bin = compute_global_metrics(Y_test, P_test, best_thr_vec)

    # Per-class 2x2 multilabel confusion (TEST)
    ml_cm = multilabel_confusion_matrix(Y_test, Yhat_test_bin, labels=list(range(C)))
    rows = []
    for i, name in enumerate(CLASSES):
        tn, fp, fn, tp = ml_cm[i].ravel().tolist()
        rows.append({
            "class": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "correct": tp + tn, "wrong": fp + fn,
            "support_pos": tp + fn, "support_neg": tn + fp,
            "predicted_pos": tp + fp, "predicted_neg": tn + fn
        })
    per_class_csv = os.path.join(RUN_DIR, "per_class_confusion_2x2_test.csv")
    pd.DataFrame(rows).to_csv(per_class_csv, index=False)

    # 5x5 CONFUSION including NONIDENTIFIER (TEST)
    def collapse_to_single(y_bin):
        single = np.full((y_bin.shape[0],), NONID_COL, dtype=int)
        ids = y_bin[:, :NONID_COL]
        for i in range(y_bin.shape[0]):
            pos = np.where(ids[i] > 0)[0]
            if len(pos) > 0:
                single[i] = int(pos[0])
        return single

    y_true_single = collapse_to_single(Y_test.astype(int))
    y_pred_single = collapse_to_single(Yhat_test_bin.astype(int))

    cm5 = np.zeros((C, C), dtype=int)
    try:
        cm5 = confusion_matrix(y_true_single, y_pred_single, labels=list(range(C)))
    except Exception:
        pass
    cm5_png = os.path.join(PLOTS_DIR, "confusion_5class_frames_test.png")
    plot_confusion_matrix(cm5, CLASSES, cm5_png, "Confusion (frame-wise, 5 classes) — TEST")
    np.savetxt(os.path.join(RUN_DIR, "confusion_5class_frames_test.csv"), cm5, fmt="%d", delimiter=",")

    # Identifiers-only (4x4)
    keep = list(range(NONID_COL))
    cm4 = cm5[np.ix_(keep, keep)] if cm5.size>0 else np.zeros((len(keep), len(keep)), dtype=int)
    cm4_png = os.path.join(PLOTS_DIR, "confusion_identifiers_only_4x4_frames_test.png")
    plot_confusion_matrix(cm4, CLASSES[:-1], cm4_png, "Confusion (identifiers only, frame-wise) — TEST")
    np.savetxt(os.path.join(RUN_DIR, "confusion_identifiers_only_4x4_frames_test.csv"), cm4, fmt="%d", delimiter=",")

    with open(os.path.join(RUN_DIR, "metrics_test.json"), "w") as f:
        json.dump({
            "classes": CLASSES,
            "val_best_thresholds": {CLASSES[i]: float(best_thr_vec[i]) for i in range(C)},
            "test": {k: float(v) for k, v in metrics.items()},
            "N_test": int(Y_test.shape[0]),
            "test_mix_matched_train": True,
            "train_mix": {CLASSES[i]: float(train_mix.get(i,0.0)) for i in range(len(CLASSES))},
            "setup": {
                "context_sec_each_side": CONTEXT_SEC,
                "frame_stride": FRAME_STRIDE,
                "attention_heads": ATTN_HEADS,
                "batch_size": BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "cache_chunk_sec": CHUNK_SEC,
                "encoder": "WavLM",
                "enc_fps": float(enc_fps),
                "frame_step_train": FRAME_STEP_TRAIN,
                "frame_step_eval": FRAME_STEP_EVAL,
                "max_frames_per_file_train": MAX_FRAMES_PER_FILE_TRAIN,
                "max_train_frames_total": MAX_TRAIN_FRAMES_TOTAL,
                "epoch_time_limit_min": EPOCH_TIME_LIMIT_MIN,
                "eval_time_limit_min": EVAL_TIME_LIMIT_MIN,
                "max_eval_samples": MAX_EVAL_SAMPLES,
                "lru_chunks": MAX_CHUNK_CACHE_ITEMS,
                "lru_label_stems": MAX_LABEL_CACHE_STEMS,
                "pin_memory": PIN_MEMORY,
                "num_workers": NUM_WORKERS,
                "prefetch_factor": PREFETCH_FACTOR,
                "persistent_workers": PERSISTENT_WORKERS
            }
        }, f, indent=2)

    tee_print("\n=== TEST (frame-wise, WavLM MH-Attn, cached; mix matched to TRAIN) ===", log_fh)
    tee_print(f"mAP (macro): {metrics.get('mAP_macro', float('nan')):.4f}", log_fh)
    tee_print(f"AUC  (macro): {metrics.get('AUC_macro', float('nan')):.4f}", log_fh)
    tee_print(f"AUC  (micro): {metrics.get('AUC_micro', float('nan')):.4f}", log_fh)
    tee_print(f"F1   (macro): {metrics.get('F1_macro', float('nan')):.4f}", log_fh)
    tee_print(f"F1   (micro): {metrics.get('F1_micro', float('nan')):.4f}", log_fh)

    tee_print("\nArtifacts:", log_fh)
    for p in [
        loss_png, per_class_csv, cm5_png, cm4_png,
        os.path.join(RUN_DIR, "metrics_val.json"),
        os.path.join(RUN_DIR, "metrics_test.json"),
        ckpt_path
    ]:
        tee_print(f" - {p}", log_fh)
    tee_print(f"Results dir: {RUN_DIR}", log_fh)
    log_fh.close()

if __name__ == "__main__":
    main()
