"""
1_Demo_Case.py — Dự đoán per-step cho một case được chọn.

Layout:
  Sidebar: chọn model .pt, dataset, top-K
  Main:    load model → encode dataset → chọn case → bảng dự đoán
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
import torch

from dataset_config import DATASET_CONFIGS, list_datasets
from loader import scan_model_dir, load_checkpoint, get_model_metadata
from pipeline import load_and_encode, predict_case

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Demo Case", page_icon="🎯", layout="wide")
st.title("🎯 Demo Case — Dự đoán sự kiện tiếp theo")

MODELS_DIR = str(_APP_DIR.parent / "output" / "models")
DEVICE = "cpu"

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang load model...")
def cached_load_model(pt_path: str):
    return load_checkpoint(pt_path, device=DEVICE)


@st.cache_data(show_spinner="Đang encode dataset (lần đầu mất ~30s)...")
def cached_encode(csv_path: str, ds_name: str, model_type: str, _config_hash: str):
    ds_config = DATASET_CONFIGS[ds_name]
    # _config_hash chỉ làm cache-key, không dùng trực tiếp
    return load_and_encode(csv_path, ds_config, st.session_state["model_config"], model_type)


def prob_bar(prob: float) -> str:
    filled = int(prob * 20)
    return "█" * filled + "░" * (20 - filled)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    # Scan model files
    model_files = scan_model_dir(MODELS_DIR)
    if not model_files:
        st.error(f"Không tìm thấy file `.pt` trong `{MODELS_DIR}`")
        st.stop()

    model_options = {m["filename"]: m for m in model_files}
    selected_file = st.selectbox(
        "📦 Chọn model (.pt)",
        list(model_options.keys()),
        format_func=lambda f: f"{f}  [{model_options[f].get('model_type','?')}  acc={model_options[f].get('test_acc') or 0:.3f}]",
    )
    selected_meta = model_options[selected_file]
    pt_path = str(Path(MODELS_DIR) / selected_file)

    st.caption(f"Model type: `{selected_meta.get('model_type','?')}`  |  epoch {selected_meta.get('epoch',0)}  |  acc {selected_meta.get('test_acc') or 0:.4f}")

    # Dataset selector
    ds_name = st.selectbox("📂 Chọn dataset", list_datasets(),
                           format_func=lambda k: DATASET_CONFIGS[k]["display_name"])

    top_k = st.slider("🏆 Top-K dự đoán", min_value=1, max_value=10, value=5)

    load_btn = st.button("🚀 Load & Encode", use_container_width=True, type="primary")

# ─── Load model ──────────────────────────────────────────────────────────────
if load_btn or "enc" in st.session_state:
    if load_btn or st.session_state.get("loaded_pt") != pt_path:
        with st.spinner("Loading model..."):
            try:
                model, config, ckpt = cached_load_model(pt_path)
                meta = get_model_metadata(ckpt)
                st.session_state["model"] = model
                st.session_state["model_config"] = config
                st.session_state["model_type"] = meta["model_type"]
                st.session_state["model_meta"] = meta
                st.session_state["loaded_pt"] = pt_path
            except Exception as e:
                st.error(f"❌ Lỗi load model: {e}")
                st.stop()

    # Encode dataset
    if load_btn or st.session_state.get("loaded_ds") != (pt_path, ds_name):
        with st.spinner("Encoding dataset..."):
            try:
                model_type = st.session_state["model_type"]
                config = st.session_state["model_config"]
                enc = cached_encode(
                    DATASET_CONFIGS[ds_name]["file"],
                    ds_name,
                    model_type,
                    str(sorted(config.items())),
                )
                st.session_state["enc"] = enc
                st.session_state["loaded_ds"] = (pt_path, ds_name)
            except Exception as e:
                st.error(f"❌ Lỗi encode dataset: {e}")
                st.stop()

# ─── Main: chọn case và hiển thị dự đoán ────────────────────────────────────
if "enc" not in st.session_state:
    st.info("👈 Chọn model và dataset rồi nhấn **Load & Encode** để bắt đầu.")
    st.stop()

enc = st.session_state["enc"]
model = st.session_state["model"]
model_type = st.session_state["model_type"]
model_config = st.session_state["model_config"]

# ─── Thông tin model ──────────────────────────────────────────────────────────
meta = st.session_state["model_meta"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", model_type.replace("Model", "").replace("Classifier", ""))
col2.metric("Test Accuracy", f"{meta['test_acc']:.2%}")
col3.metric("Số cases", f"{len(enc['case_ids']):,}")
col4.metric("Top-K", top_k)

st.divider()

# ─── Chọn case ───────────────────────────────────────────────────────────────
case_ids = enc["case_ids"]
selected_case = st.selectbox(
    "🔍 Chọn Case ID",
    case_ids,
    help="ID của case trong dataset để xem dự đoán per-step",
)

if selected_case:
    # Hiển thị event history
    event_history = enc["case_events"].get(selected_case, [])
    with st.expander(f"📋 Lịch sử sự kiện — Case `{selected_case}` ({len(event_history)} events)", expanded=True):
        case_index = DATASET_CONFIGS[ds_name]["case_index"]
        display_cols = DATASET_CONFIGS[ds_name]["display_cols"]
        event_df = enc.get("event_df")
        if event_df is not None:
            case_rows = event_df[event_df[case_index] == selected_case][display_cols].reset_index(drop=True)
            st.dataframe(case_rows, use_container_width=True, height=200)
        else:
            st.write(event_history)

    # Chạy inference
    with st.spinner("Đang dự đoán..."):
        try:
            result = predict_case(
                enc, selected_case, model, model_type, model_config,
                top_k=top_k, device=DEVICE,
            )
        except Exception as e:
            st.error(f"❌ Lỗi inference: {e}")
            st.stop()

    preds = result["predictions_per_step"]
    st.subheader(f"🔮 Dự đoán per-step — {len(preds)} bước")

    # Build display dataframe
    rows = []
    for p in preds:
        top_event = p["top_k"][0]["event"] if p["top_k"] else "—"
        top_prob  = p["top_k"][0]["probability"] if p["top_k"] else 0.0
        gt        = p["ground_truth"] or "—"
        correct   = "✅" if p["ground_truth"] and top_event == p["ground_truth"] else "❌"

        top_k_str = "  |  ".join(
            f"{t['event']} ({t['probability']:.2%})" for t in p["top_k"]
        )
        rows.append({
            "Bước": p["step"],
            "Sự kiện hiện tại": p["context_event"],
            "Dự đoán #1": f"{top_event} ({top_prob:.2%})",
            f"Top-{top_k}": top_k_str,
            "Ground Truth": gt,
            "✓/✗": correct,
        })

    df_pred = pd.DataFrame(rows)
    st.dataframe(
        df_pred,
        use_container_width=True,
        height=min(40 * len(rows) + 40, 600),
        column_config={
            "Bước": st.column_config.NumberColumn(width="small"),
            "✓/✗": st.column_config.TextColumn(width="small"),
        },
    )

    # Accuracy per case
    total = len(preds)
    correct_cnt = sum(1 for p in preds if p["top_k"] and p["ground_truth"] and p["top_k"][0]["event"] == p["ground_truth"])
    topk_correct = sum(
        1 for p in preds
        if p["ground_truth"] and any(t["event"] == p["ground_truth"] for t in p["top_k"])
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Top-1 Accuracy (case)", f"{correct_cnt / max(total,1):.2%}")
    c2.metric(f"Top-{top_k} Accuracy (case)", f"{topk_correct / max(total,1):.2%}")
    c3.metric("Số bước dự đoán", total)

    # Probability bar chart for selected step
    st.divider()
    st.subheader("📊 Phân phối xác suất — chọn bước")
    step_idx = st.slider("Bước", min_value=0, max_value=len(preds) - 1, value=0)
    sel_step = preds[step_idx]
    chart_data = pd.DataFrame(sel_step["top_k"]).rename(
        columns={"event": "Sự kiện", "probability": "Xác suất"}
    )
    st.bar_chart(chart_data.set_index("Sự kiện")["Xác suất"])
    if sel_step["ground_truth"]:
        st.caption(f"Ground truth bước {step_idx}: **{sel_step['ground_truth']}**")
