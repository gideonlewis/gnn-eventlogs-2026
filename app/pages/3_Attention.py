"""
3_Attention.py — Attention & Time-Decay analysis cho DualGATTimeAwareModel và DualGATTimeAwareETModel.

Hiển thị:
  - Attention weights per edge cho một case được chọn
  - Time-decay weights
  - Edge-type scores (DualGATTimeAwareETModel)
  - Heatmap attention theo bước
"""

import sys
import os
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent
_SRC_DIR = _APP_DIR.parent / "src"
for d in (_APP_DIR / "core", _SRC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Batch

from dataset_config import DATASET_CONFIGS, list_datasets
from loader import scan_model_dir, load_checkpoint, get_model_metadata
from pipeline import load_and_encode, _build_gat_graph_single

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Attention Analysis", page_icon="🧠", layout="wide")
st.title("🧠 Attention Analysis — Time-Decay & Edge Weights")

MODELS_DIR = str(_APP_DIR.parent / "output" / "models")
DEVICE = "cpu"

ATTENTION_MODELS = {"DualGATTimeAwareModel", "DualGATTimeAwareETModel"}

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang load model...")
def cached_load_model(pt_path: str):
    return load_checkpoint(pt_path, device=DEVICE)


@st.cache_data(show_spinner="Encoding dataset...")
def cached_encode(csv_path: str, ds_name: str, model_type: str, _cfg_hash: str):
    from pipeline import load_and_encode as _le
    ds_config = DATASET_CONFIGS[ds_name]
    config = st.session_state.get("attn_model_config", {})
    return _le(csv_path, ds_config, config, model_type)


def _run_attn_inference(enc, case_idx, model, model_type):
    """Chạy forward với return_attention=True. Trả về (output, attn_data)."""
    data_list = _build_gat_graph_single(enc, case_idx, model_type)
    batch = Batch.from_data_list(data_list).to(DEVICE)
    model.eval()
    with torch.no_grad():
        result = model(batch, return_attention=True)
    if isinstance(result, tuple):
        output, attn_data = result
    else:
        output, attn_data = result, None
    return output, attn_data


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    model_files = scan_model_dir(MODELS_DIR)
    if not model_files:
        st.error(f"Không tìm thấy file `.pt` trong `{MODELS_DIR}`")
        st.stop()

    # Lọc chỉ hiển thị Time-Aware models
    attn_files = [m for m in model_files if m["model_type"] in ATTENTION_MODELS]
    all_files = model_files  # cho phép dùng tất cả

    show_all = st.checkbox("Hiển thị tất cả models", value=False,
                           help="Tắt để chỉ hiện Time-Aware models có attention")
    display_files = all_files if show_all else (attn_files if attn_files else all_files)

    if not display_files:
        st.warning("Chưa có file .pt nào.")
        st.stop()

    model_options = {m["filename"]: m for m in display_files}
    selected_file = st.selectbox(
        "📦 Chọn model",
        list(model_options.keys()),
        format_func=lambda f: f"{f}  [{model_options[f]['model_type']}]",
    )
    pt_path = str(Path(MODELS_DIR) / selected_file)
    sel_model_type = model_options[selected_file]["model_type"]

    if sel_model_type not in ATTENTION_MODELS:
        st.warning(f"⚠️ `{sel_model_type}` không có attention output. Kết quả chỉ hiển thị logits.")

    ds_name = st.selectbox("📂 Chọn dataset", list_datasets(),
                           format_func=lambda k: DATASET_CONFIGS[k]["display_name"])

    load_btn = st.button("🚀 Load & Encode", use_container_width=True, type="primary")

# ─── Load model & encode ─────────────────────────────────────────────────────
if load_btn or "attn_enc" in st.session_state:
    if load_btn or st.session_state.get("attn_loaded_pt") != pt_path:
        with st.spinner("Loading model..."):
            try:
                model, config, ckpt = cached_load_model(pt_path)
                meta = get_model_metadata(ckpt)
                model_type = meta["model_type"]
                st.session_state["attn_model"] = model
                st.session_state["attn_model_config"] = config
                st.session_state["attn_model_type"] = model_type
                st.session_state["attn_meta"] = meta
                st.session_state["attn_loaded_pt"] = pt_path
            except Exception as e:
                st.error(f"❌ Lỗi load model: {e}")
                st.stop()

    if load_btn or st.session_state.get("attn_loaded_ds") != (pt_path, ds_name):
        with st.spinner("Encoding dataset..."):
            try:
                model_type = st.session_state["attn_model_type"]
                config = st.session_state["attn_model_config"]
                enc = cached_encode(
                    DATASET_CONFIGS[ds_name]["file"],
                    ds_name,
                    model_type,
                    str(sorted(config.items())),
                )
                st.session_state["attn_enc"] = enc
                st.session_state["attn_loaded_ds"] = (pt_path, ds_name)
            except Exception as e:
                st.error(f"❌ Lỗi encode dataset: {e}")
                st.stop()

# ─── Main ────────────────────────────────────────────────────────────────────
if "attn_enc" not in st.session_state:
    st.info("👈 Chọn model và dataset rồi nhấn **Load & Encode** để bắt đầu.")
    st.stop()

enc = st.session_state["attn_enc"]
model = st.session_state["attn_model"]
model_type = st.session_state["attn_model_type"]
meta = st.session_state["attn_meta"]

if enc.get("pipeline_type") == "prefix":
    st.warning("PrefixGCNClassifier không có per-edge attention. Chọn một GAT model.")
    st.stop()

# Thông số
c1, c2, c3 = st.columns(3)
c1.metric("Model", model_type)
c2.metric("Test Accuracy", f"{meta.get('test_acc', 0):.2%}")
c3.metric("Has Attention", "✅" if model_type in ATTENTION_MODELS else "❌")

st.divider()

# ─── Chọn case ───────────────────────────────────────────────────────────────
case_ids = enc["case_ids"]
le_event = enc["le_event"]

# Lọc case có đủ độ dài để thú vị
min_len = st.slider("Độ dài case tối thiểu", 3, 20, 5)
filtered_cases = [
    cid for cid in case_ids
    if len(enc["case_events"].get(cid, [])) >= min_len
]
if not filtered_cases:
    st.warning(f"Không có case nào có >= {min_len} events.")
    st.stop()

selected_case = st.selectbox(
    "🔍 Chọn Case ID",
    filtered_cases,
    format_func=lambda c: f"{c}  ({len(enc['case_events'].get(c, []))} events)",
)

case_idx = case_ids.index(selected_case)
event_names = enc["case_events"].get(selected_case, [])
n_events = len(event_names)
n_edges = n_events - 1

st.markdown(f"**Case `{selected_case}`** — {n_events} events, {n_edges} edges")
with st.expander("📋 Event sequence", expanded=False):
    st.write(" → ".join(event_names))

# ─── Inference with attention ────────────────────────────────────────────────
with st.spinner("Chạy inference với attention..."):
    try:
        output, attn_data = _run_attn_inference(enc, case_idx, model, model_type)
    except Exception as e:
        st.error(f"❌ Lỗi inference: {e}")
        st.stop()

# ─── Logits visualization (all model types) ──────────────────────────────────
st.subheader("🔮 Dự đoán per-step (Logits)")

pred_rows = []
for step in range(output.shape[0]):
    logits = output[step]
    top3 = logits.topk(min(3, logits.size(0)))
    preds_str = ", ".join(
        f"{le_event.inverse_transform([i.item()])[0] if i.item() < len(le_event.classes_) else f'idx_{i.item()}'} ({p:.2%})"
        for p, i in zip(torch.softmax(top3.values.float(), dim=0).tolist(), top3.indices.tolist())
    )
    pred_rows.append({"Bước": step, "Sự kiện": event_names[step] if step < n_events else "?", "Top-3": preds_str})

st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, height=min(35 * len(pred_rows) + 40, 400))

# ─── Attention visualization ─────────────────────────────────────────────────
if attn_data is None or model_type not in ATTENTION_MODELS:
    st.info("Model này không trả về attention weights.")
    st.stop()

st.divider()
st.subheader("🔥 Attention Weights")

# attn_data là dict: alpha_embed, alpha_event, alpha_final (+ decay_*, edge_index, time)
edge_index = attn_data.get("edge_index")  # [2, E] cpu tensor

if edge_index is None or edge_index.shape[1] == 0:
    st.info("Case này chỉ có 1 event, không có edge để hiển thị.")
    st.stop()

E = edge_index.shape[1]
edge_labels = []
for e in range(E):
    src = int(edge_index[0, e])
    tgt = int(edge_index[1, e])
    src_name = event_names[src] if src < n_events else f"node_{src}"
    tgt_name = event_names[tgt] if tgt < n_events else f"node_{tgt}"
    edge_labels.append(f"{src_name}→{tgt_name}")

# Chọn loại attention để hiển thị
attn_type = st.selectbox(
    "Loại attention",
    ["alpha_embed", "alpha_event", "alpha_final"],
    format_func=lambda x: {
        "alpha_embed": "Embedding path",
        "alpha_event": "Event feature path",
        "alpha_final": "Concat path (final)",
    }[x],
)

alpha_tensor = attn_data.get(attn_type)  # [E, H]
if alpha_tensor is None:
    st.warning(f"Không có dữ liệu cho `{attn_type}`.")
else:
    if alpha_tensor.dim() == 1:
        alpha_tensor = alpha_tensor.unsqueeze(1)
    num_heads = alpha_tensor.shape[1]
    # Average across heads
    alpha_mean = alpha_tensor.float().mean(dim=1).numpy()  # [E]
    # Normalize 0-1
    alpha_norm = (alpha_mean - alpha_mean.min()) / (alpha_mean.max() - alpha_mean.min() + 1e-8)

    df_alpha = pd.DataFrame({
        "Edge": edge_labels,
        "Attention (avg heads)": alpha_norm,
    }).set_index("Edge")

    st.bar_chart(df_alpha)

    # Table with per-head breakdown
    with st.expander("Per-head attention weights", expanded=False):
        head_data = {"Edge": edge_labels}
        for h in range(num_heads):
            vals = alpha_tensor[:, h].float().numpy()
            vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
            head_data[f"Head {h}"] = vals_norm
        st.dataframe(pd.DataFrame(head_data), use_container_width=True)

# ─── Time-Decay weights ──────────────────────────────────────────────────────
st.divider()
st.subheader("⏱️ Time-Decay Weights")

decay_map = {
    "decay_embed": "Embedding path",
    "decay_event": "Event feature path",
    "decay_final": "Concat path",
}

decay_found = False
for decay_key, decay_label in decay_map.items():
    decay_tensor = attn_data.get(decay_key)
    if decay_tensor is None:
        continue
    decay_found = True
    if decay_tensor.dim() > 1:
        decay_vals = decay_tensor.float().squeeze().mean(dim=-1).numpy()
    else:
        decay_vals = decay_tensor.float().numpy()
    if len(decay_vals) == E:
        df_d = pd.DataFrame({"Edge": edge_labels, decay_label: decay_vals}).set_index("Edge")
        st.bar_chart(df_d)

if not decay_found:
    st.info("Không có decay weights (model chưa lưu _decay).")

# ─── Edge-type analysis (DualGATTimeAwareETModel) ────────────────────────────
if model_type == "DualGATTimeAwareETModel":
    st.divider()
    st.subheader("🔗 Edge-Type Scores")

    et_map = {
        "edge_type_score_embed": "Embedding path",
        "edge_type_score_event": "Event feature path",
        "edge_type_score_final": "Concat path",
    }
    et_found = False
    for et_key, et_label in et_map.items():
        et_tensor = attn_data.get(et_key)
        if et_tensor is None:
            continue
        et_found = True
        if et_tensor.dim() > 1:
            et_vals = et_tensor.float().mean(dim=-1).numpy()
        else:
            et_vals = et_tensor.float().numpy()
        if len(et_vals) == E:
            df_et = pd.DataFrame({"Edge": edge_labels, et_label: et_vals}).set_index("Edge")
            st.bar_chart(df_et)

    if not et_found:
        st.info("Không có edge-type scores.")

# ─── Attention heatmap (bước × sự kiện) ─────────────────────────────────────
st.divider()
st.subheader("🗺️ Attention Heatmap — Node × Step")

if alpha_tensor is not None and n_events >= 2:
    # Tạo attention matrix [nodes, nodes] dạng adj với e→target
    attn_matrix = np.zeros((n_events, n_events))
    for e in range(E):
        src = int(edge_index[0, e])
        tgt = int(edge_index[1, e])
        if src < n_events and tgt < n_events:
            attn_matrix[src, tgt] = float(alpha_mean[e]) if e < len(alpha_mean) else 0.0

    df_heat = pd.DataFrame(
        attn_matrix,
        index=[f"{i}: {event_names[i]}" if i < n_events else str(i) for i in range(n_events)],
        columns=[f"{i}: {event_names[i]}" if i < n_events else str(i) for i in range(n_events)],
    )
    st.dataframe(
        df_heat.style.background_gradient(cmap="YlOrRd", axis=None),
        use_container_width=True,
        height=min(35 * n_events + 60, 500),
    )
    st.caption("Giá trị attention từ node nguồn (hàng) đến node đích (cột).")
