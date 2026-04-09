"""
pipeline.py — Inference pipeline cho 5 model types.

Public API:
    load_and_encode(csv_path, ds_config, model_config, model_type)
        → dict  (EncodedResult, cacheable)

    predict_case(enc, case_id, model, model_type, model_config,
                 top_k, device, return_attention)
        → dict  (PredictResult)

Mỗi model type dùng hàm prepare_data khác nhau:

    DualGATModel           → GATConv.prepare_data_core_timedif   (edge_attr = time_diffs, node_times)
    DualGATTimeAwareModel  → GATConvTimeDecay.prepare_data_core  (graph.time = node_times, NO edge_attr)
    DualGAT2EdgesModel     → GATConvStatusEmb.prepare_data_core_2edges   (edge_type + edge_time_diff)
    DualGATTimeAwareETModel→ GATConvTimeDecayStatusEmb.prepare_data_core_2edges (+ node_times)
    PrefixGCNClassifier    → PrefixEmbeddingGCN.prepare_data     (fixed prefix, graph-level pred)
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch

# ── Path setup ───────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Suppress PyG / TF import warnings ────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

from DataEncoder import (
    encode_label_event,
    encode_pad_event,
    encode_pad_sequence,
    scale_time_differences_fast_fixed,
    node_time_list,
    event_transition_edge,
    encode_event_prefix_label,
)
from GATConv import prepare_data_core_timedif
from GATConvTimeDecay import prepare_data_core as _prepare_core_td
from GATConvStatusEmb import prepare_data_core_2edges as _prepare_2e_se
from GATConvTimeDecayStatusEmb import prepare_data_core_2edges as _prepare_2e_tdet
from PrefixEmbeddingGCN import prepare_data as _prepare_prefix


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _decode_top_k(logits_1d: torch.Tensor, le_event, top_k: int) -> list:
    """Given a 1-D logit tensor, return top-k dicts {event, probability, class_idx}."""
    probs = torch.softmax(logits_1d.float(), dim=0)
    k = min(top_k, probs.size(0))
    top_vals, top_idx = torch.topk(probs, k)
    results = []
    for prob, idx in zip(top_vals.tolist(), top_idx.tolist()):
        try:
            name = le_event.inverse_transform([idx])[0]
        except Exception:
            name = f"class_{idx}"
        results.append({"event": name, "probability": round(prob, 4), "class_idx": idx})
    return results


def _build_gat_graph_single(enc: dict, case_idx: int, model_type: str) -> list:
    """
    Build a 1-element PyG Data list for the case at position case_idx.
    Dispatches to the correct prepare_data_* function based on model_type.
    """
    cf = enc["combined_features"][case_idx : case_idx + 1]
    ce = enc["core_encode"][case_idx : case_idx + 1]
    td = [enc["scaled_time_diffs"][case_idx]]
    nt = [enc["node_times"][case_idx]] if enc.get("node_times") is not None else None
    et = (
        [enc["edge_types_encoded"][case_idx]]
        if enc.get("edge_types_encoded") is not None
        else None
    )

    if model_type == "DualGATModel":
        return prepare_data_core_timedif(cf, ce, td, nt)
    elif model_type == "DualGATTimeAwareModel":
        return _prepare_core_td(cf, ce, td, nt)
    elif model_type == "DualGAT2EdgesModel":
        return _prepare_2e_se(cf, ce, td, et)
    elif model_type == "DualGATTimeAwareETModel":
        return _prepare_2e_tdet(cf, ce, td, et, nt)
    else:
        raise ValueError(f"Unknown GAT model_type: {model_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# load_and_encode
# ─────────────────────────────────────────────────────────────────────────────

_APP_DIR = Path(__file__).resolve().parent.parent   # .../app/
_PROJECT_ROOT = _APP_DIR.parent                      # project root


def _resolve_csv_path(csv_path: str) -> str:
    """
    Resolve csv_path có thể là:
      - absolute path (dùng luôn)
      - path tương đối từ project root (e.g. 'output/helpdesk.csv')
      - path bắt đầu bằng '../output/' (relative từ app/ dir)
    """
    p = Path(csv_path)
    if p.is_absolute() and p.exists():
        return str(p)
    # Thử resolve từ project root trước
    candidate = _PROJECT_ROOT / csv_path
    if candidate.exists():
        return str(candidate)
    # Thử resolve từ app/ dir (dành cho '../output/...' patterns)
    candidate2 = (_APP_DIR / csv_path).resolve()
    if candidate2.exists():
        return str(candidate2)
    # Fallback: trả lại nguyên bản, pandas sẽ raise lỗi rõ ràng
    return csv_path


def load_and_encode(csv_path: str, ds_config: dict, model_config: dict, model_type: str) -> dict:
    """
    Load CSV và chạy toàn bộ encoding pipeline.

    Returns một dict (EncodedResult) chứa tất cả arrays đã encode + le_event.
    Dict này có thể được cache ở Streamlit với @st.cache_data.
    """
    event = pd.read_csv(_resolve_csv_path(csv_path))
    case_index = ds_config["case_index"]
    core_event = ds_config["core_event"]
    time_col = ds_config["time_col"]

    # Ép ec1 sang str nếu cần (BPI12, BPI12w)
    if ds_config.get("ec1_as_str"):
        event["ec1"] = event["ec1"].astype(str)

    # Lọc case quá ngắn
    min_sz = ds_config.get("min_seq_size", 2)
    event = event[
        event.groupby(case_index)[case_index].transform("size") >= min_sz
    ].reset_index(drop=True)

    if model_type == "PrefixGCNClassifier":
        return _encode_prefix(event, ds_config, model_config, case_index, core_event, time_col)
    else:
        # Build sequence DataFrame (1 row per unique case) for GAT models
        seq_cols = ds_config["seq_cols"]
        sequence = (
            event[seq_cols]
            .drop_duplicates(subset=[case_index])
            .reset_index(drop=True)
        )
        return _encode_gat(event, sequence, ds_config, model_config, model_type, case_index, core_event, time_col)


# ─── GAT branch ──────────────────────────────────────────────────────────────

def _encode_gat(event, sequence, ds_config, model_config, model_type, case_index, core_event, time_col):
    cat_col_event = ds_config["cat_col_event"]
    num_col_event = ds_config["num_col_event"]
    cat_col_seq = ds_config["cat_col_seq"]
    num_col_seq = ds_config["num_col_seq"]

    # 1. Label encode (next-event targets)
    core_encode, y_encode, core_size, output_size, le_event = encode_label_event(
        event, core_event, case_index
    )

    # 2. Encode & pad event-level features
    event_encode = encode_pad_event(
        event, cat_col_event, num_col_event, case_index,
        cat_mask=True, num_mask=True, eos=False,
    )

    # 3. Encode sequence-level features (1 row per case)
    sequence_encode = encode_pad_sequence(sequence, cat_col_seq, num_col_seq)

    # 4. Scale time differences
    scaled_time_diffs = scale_time_differences_fast_fixed(
        event, sequence, time_col, case_index
    )

    # 5. Node times for models with time-aware attention
    node_times = None
    if model_type in ("DualGATModel", "DualGATTimeAwareModel", "DualGATTimeAwareETModel"):
        node_times = node_time_list(event, time_col, case_index)

    # 6. Edge transition types for 2-edge models
    edge_types_encoded = None
    trans_size = None
    if model_type in ("DualGAT2EdgesModel", "DualGATTimeAwareETModel"):
        status_col = ds_config.get("status_col", core_event)
        edge_types_encoded, _le_edge, trans_size = event_transition_edge(
            event, sequence, status_col, case_index
        )

    # 7. Expand sequence feats → [num_cases, max_events, seq_feat_dim] then concat
    max_num_events = event_encode.shape[1]
    seq_exp = np.expand_dims(sequence_encode, axis=1)
    seq_exp = np.repeat(seq_exp, max_num_events, axis=1)
    combined_features = np.concatenate((event_encode, seq_exp), axis=2)

    # 8. Per-case event name list for display
    case_events: dict = {}
    for cid, grp in event.groupby(case_index, sort=False):
        case_events[cid] = grp[core_event].tolist()

    case_ids = sequence[case_index].tolist()

    return {
        "pipeline_type": "gat",
        "model_type": model_type,
        "case_ids": case_ids,
        "core_encode": core_encode,
        "y_encode": y_encode,
        "combined_features": combined_features,
        "scaled_time_diffs": scaled_time_diffs,
        "node_times": node_times,
        "edge_types_encoded": edge_types_encoded,
        "trans_size": trans_size,
        "le_event": le_event,
        "output_size": output_size,
        "case_events": case_events,
        "event_df": event,
    }


# ─── Prefix branch ───────────────────────────────────────────────────────────

def _encode_prefix(event, ds_config, model_config, case_index, core_event, time_col):
    cat_col_event = ds_config["cat_col_event"]
    num_col_event = ds_config["num_col_event"]
    cat_col_seq = ds_config["cat_col_seq"]
    num_col_seq = ds_config["num_col_seq"]

    prefix_size = model_config.get("prefix_size", 10)

    # Chỉ giữ các case có đủ ít nhất prefix_size events
    event = event[
        event.groupby(case_index)[case_index].transform("size") >= prefix_size
    ].reset_index(drop=True)

    # encode_event_prefix_label → sliding window subsequences
    # Returns: text_encode(core_encode), event_encode(features), y_encode, text_size, output_dim
    text_encode, event_encode, y_encode, text_size, output_size = encode_event_prefix_label(
        event, core_event, cat_col_event, num_col_event, case_index, prefix_size,
        cat_mask=False, num_mask=False,
    )

    # Reconstruct le_event để decode predictions
    le_event = LabelEncoder()
    le_event.fit(sorted(set(event[core_event].tolist())) + ["EOS"])

    # "pred_sequence": một row per prediction instance (cumcount >= prefix_size-1)
    pred_sequence = event[
        event.groupby(case_index).cumcount() >= prefix_size - 1
    ].reset_index(drop=True)

    # Sequence-level features (aligned to pred_sequence rows)
    sequence_encode = encode_pad_sequence(pred_sequence, cat_col_seq, num_col_seq)

    # Time diffs (per prediction instance → maps to full-case diffs)
    scaled_time_diffs = scale_time_differences_fast_fixed(
        event, pred_sequence, time_col, case_index
    )

    # Build PyG Data list (1 per prediction instance)
    event_feature_list = _prepare_prefix(event_encode, text_encode, scaled_time_diffs)

    # Per-case event name list
    case_events: dict = {}
    for cid, grp in event.groupby(case_index, sort=False):
        case_events[cid] = grp[core_event].tolist()

    return {
        "pipeline_type": "prefix",
        "model_type": "PrefixGCNClassifier",
        "case_ids": list(pred_sequence[case_index].unique()),
        "text_encode": text_encode,
        "event_encode": event_encode,
        "y_encode": y_encode,
        "sequence_encode": sequence_encode,
        "scaled_time_diffs": scaled_time_diffs,
        "event_feature_list": event_feature_list,
        "pred_sequence": pred_sequence,
        "le_event": le_event,
        "output_size": output_size,
        "prefix_size": prefix_size,
        "case_events": case_events,
        "event_df": event,
    }


# ─────────────────────────────────────────────────────────────────────────────
# predict_case
# ─────────────────────────────────────────────────────────────────────────────

def predict_case(
    enc: dict,
    case_id,
    model,
    model_type: str,
    model_config: dict,
    top_k: int = 5,
    device: str = "cpu",
    return_attention: bool = False,
) -> dict:
    """
    Chạy inference cho một case và trả về dự đoán per-step.

    Returns:
        {
            "case_id": ...,
            "event_history": ["EventA", "EventB", ...],
            "predictions_per_step": [
                {
                    "step": 0,
                    "context_event": "EventA",
                    "top_k": [{"event": ..., "probability": ..., "class_idx": ...}, ...],
                    "ground_truth": "EventB"   # None nếu là bước cuối (EOS)
                },
                ...
            ],
            "attention": {...}   # chỉ có khi return_attention=True và model hỗ trợ
        }
    """
    if enc["pipeline_type"] == "gat":
        return _predict_gat(enc, case_id, model, model_type, top_k, device, return_attention)
    else:
        return _predict_prefix(enc, case_id, model, model_config, top_k, device)


# ─── GAT prediction ──────────────────────────────────────────────────────────

def _predict_gat(enc, case_id, model, model_type, top_k, device, return_attention):
    case_ids = enc["case_ids"]
    if case_id not in case_ids:
        raise ValueError(f"case_id={case_id!r} không tìm thấy trong encoded dataset.")
    case_idx = case_ids.index(case_id)

    # Build graph (batch of 1)
    data_list = _build_gat_graph_single(enc, case_idx, model_type)
    batch = Batch.from_data_list(data_list).to(device)

    model.eval()
    attn_data = None
    has_attn_support = model_type in ("DualGATTimeAwareModel", "DualGATTimeAwareETModel")

    with torch.no_grad():
        if has_attn_support and return_attention:
            result = model(batch, return_attention=True)
            output, attn_data = result if isinstance(result, tuple) else (result, None)
        else:
            output = model(batch)

    # output shape: [num_events_in_case, output_dim]
    num_events = output.shape[0]
    le_event = enc["le_event"]
    y_encode = enc["y_encode"]   # shape [num_cases, max_seq_len, 1]
    event_names = enc["case_events"].get(case_id, [])

    predictions_per_step = []
    for step in range(num_events):
        # output[step] → predicts the NEXT event after position `step`
        logits = output[step]
        top_k_preds = _decode_top_k(logits, le_event, top_k)

        context_event = event_names[step] if step < len(event_names) else "?"

        # Ground truth from y_encode
        gt_event = None
        try:
            raw = y_encode[case_idx][step]
            gt_idx = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
            if gt_idx >= 0:
                gt_event = le_event.inverse_transform([gt_idx])[0]
        except Exception:
            pass

        predictions_per_step.append({
            "step": step,
            "context_event": context_event,
            "top_k": top_k_preds,
            "ground_truth": gt_event,
        })

    result = {
        "case_id": case_id,
        "event_history": event_names,
        "predictions_per_step": predictions_per_step,
    }
    if attn_data is not None:
        result["attention"] = attn_data
    return result


# ─── Prefix prediction ───────────────────────────────────────────────────────

def _predict_prefix(enc, case_id, model, model_config, top_k, device):
    prefix_size = enc["prefix_size"]
    le_event = enc["le_event"]
    case_events = enc["case_events"].get(case_id, [])

    if len(case_events) < prefix_size:
        raise ValueError(
            f"Case {case_id!r} chỉ có {len(case_events)} events, cần ít nhất {prefix_size}."
        )

    # Tìm prediction instances của case này trong pred_sequence
    pred_seq = enc["pred_sequence"]
    case_index = pred_seq.columns[0]  # first column is case_index (always 'sequence')
    # Get case_index column name properly
    case_index_col = [c for c in pred_seq.columns if c == "sequence"][0]
    case_rows = pred_seq[pred_seq[case_index_col] == case_id]

    if len(case_rows) == 0:
        raise ValueError(f"Không tìm thấy prediction instance nào cho case {case_id!r}.")

    # Dùng instance cuối cùng (dự đoán sau đoạn prefix gần nhất)
    base_offset = pred_seq.index[0]
    last_iloc = case_rows.index[-1] - base_offset
    last_iloc = int(min(last_iloc, len(enc["event_feature_list"]) - 1))

    # Build batch of 1
    data = enc["event_feature_list"][last_iloc]
    seq_feat = torch.tensor(
        enc["sequence_encode"][last_iloc : last_iloc + 1], dtype=torch.float32
    ).to(device)
    batch = Batch.from_data_list([data]).to(device)

    model.eval()
    with torch.no_grad():
        output = model(batch, seq_feat)   # [1, output_dim]

    logits = output[0]   # [output_dim]
    top_k_preds = _decode_top_k(logits, le_event, top_k)

    # Ground truth
    gt_event = None
    try:
        gt_idx = int(enc["y_encode"][last_iloc])
        if gt_idx >= 0:
            gt_event = le_event.inverse_transform([gt_idx])[0]
    except Exception:
        pass

    # Context = last prefix_size events (or all if fewer)
    context_events = case_events[-prefix_size:]

    predictions_per_step = [{
        "step": prefix_size - 1,
        "context_event": context_events[-1] if context_events else "?",
        "top_k": top_k_preds,
        "ground_truth": gt_event,
    }]

    return {
        "case_id": case_id,
        "event_history": context_events,
        "predictions_per_step": predictions_per_step,
    }
