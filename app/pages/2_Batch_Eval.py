"""
2_Batch_Eval.py — Đánh giá toàn bộ dataset.

- Top-1 / Top-K accuracy
- Per-class classification report
- Confusion matrix
- Phân phối độ dài sequence
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
from pipeline import load_and_encode

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Batch Evaluation", page_icon="📊", layout="wide")
st.title("📊 Batch Evaluation — Đánh giá toàn bộ dataset")

MODELS_DIR = str(_APP_DIR.parent / "output" / "models")
DEVICE = "cpu"

# ─── Helpers ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Đang load model...")
def cached_load_model(pt_path: str):
    return load_checkpoint(pt_path, device=DEVICE)


@st.cache_data(show_spinner="Encoding dataset...")
def cached_encode(csv_path: str, ds_name: str, model_type: str, _cfg_hash: str):
    from pipeline import load_and_encode as _le
    ds_config = DATASET_CONFIGS[ds_name]
    model_config = st.session_state.get("be_model_config", {})
    return _le(csv_path, ds_config, model_config, model_type)


def _run_gat_batch_eval(enc, model, model_type, top_k, device):
    """Chạy inference cả dataset cho GAT models. Trả về (all_preds, all_gt, all_topk, all_probs)."""
    from pipeline import _build_gat_graph_single

    model.eval()
    case_ids = enc["case_ids"]
    le_event = enc["le_event"]
    y_encode = enc["y_encode"]

    all_preds, all_gt, all_topk_correct = [], [], []

    for idx, cid in enumerate(case_ids):
        data_list = _build_gat_graph_single(enc, idx, model_type)
        batch = Batch.from_data_list(data_list).to(device)

        with torch.no_grad():
            output = model(batch)  # [num_events, output_dim]

        num_events = output.shape[0]
        for step in range(num_events):
            logits = output[step]
            pred_idx = int(logits.argmax().item())
            try:
                raw = y_encode[idx][step]
                gt_idx = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
            except Exception:
                gt_idx = -1

            if gt_idx < 0:
                continue

            all_preds.append(pred_idx)
            all_gt.append(gt_idx)

            top_k_idx = logits.topk(min(top_k, logits.size(0))).indices.tolist()
            all_topk_correct.append(int(gt_idx in top_k_idx))

    return all_preds, all_gt, all_topk_correct


def _run_prefix_batch_eval(enc, model, top_k, device):
    """Chạy inference cả dataset cho PrefixGCN. Trả về (all_preds, all_gt, all_topk_correct)."""
    model.eval()
    event_feature_list = enc["event_feature_list"]
    sequence_encode = enc["sequence_encode"]
    y_encode = enc["y_encode"]
    n = len(event_feature_list)

    all_preds, all_gt, all_topk_correct = [], [], []

    BATCH_SZ = 256
    for start in range(0, n, BATCH_SZ):
        end = min(start + BATCH_SZ, n)
        batch_data = Batch.from_data_list(event_feature_list[start:end]).to(device)
        seq_feat = torch.tensor(sequence_encode[start:end], dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(batch_data, seq_feat)  # [batch, output_dim]

        for i in range(end - start):
            logits = output[i]
            pred_idx = int(logits.argmax().item())
            gt_idx = int(y_encode[start + i])

            if gt_idx < 0:
                continue

            all_preds.append(pred_idx)
            all_gt.append(gt_idx)
            top_k_idx = logits.topk(min(top_k, logits.size(0))).indices.tolist()
            all_topk_correct.append(int(gt_idx in top_k_idx))

    return all_preds, all_gt, all_topk_correct


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    model_files = scan_model_dir(MODELS_DIR)
    if not model_files:
        st.error(f"Không tìm thấy file `.pt` trong `{MODELS_DIR}`")
        st.stop()

    model_options = {m["filename"]: m for m in model_files}
    selected_file = st.selectbox(
        "📦 Chọn model",
        list(model_options.keys()),
        format_func=lambda f: f"{f}  [{model_options[f].get('model_type','?')}  acc={model_options[f].get('test_acc') or 0:.3f}]",
    )
    pt_path = str(Path(MODELS_DIR) / selected_file)

    ds_name = st.selectbox("📂 Chọn dataset", list_datasets(),
                           format_func=lambda k: DATASET_CONFIGS[k]["display_name"])

    top_k = st.slider("🏆 Top-K", 1, 10, 3)

    sample_pct = st.slider(
        "📉 Lấy mẫu (%)",
        min_value=5, max_value=100, value=100, step=5,
        help="Giảm % để eval nhanh hơn",
    )

    eval_btn = st.button("▶️ Chạy Evaluation", use_container_width=True, type="primary")

# ─── Main ────────────────────────────────────────────────────────────────────
if not eval_btn and "be_results" not in st.session_state:
    st.info("👈 Chọn model và dataset rồi nhấn **Chạy Evaluation**.")
    st.stop()

if eval_btn:
    # Load model
    with st.spinner("Loading model..."):
        try:
            model, config, ckpt = cached_load_model(pt_path)
            meta = get_model_metadata(ckpt)
            model_type = meta["model_type"]
            st.session_state["be_model_config"] = config
        except Exception as e:
            st.error(f"❌ Lỗi load model: {e}")
            st.stop()

    # Encode dataset
    with st.spinner("Encoding dataset..."):
        try:
            enc = cached_encode(
                DATASET_CONFIGS[ds_name]["file"],
                ds_name,
                model_type,
                str(sorted(config.items())),
            )
        except Exception as e:
            st.error(f"❌ Lỗi encode: {e}")
            st.stop()

    # Sample nếu cần
    if sample_pct < 100:
        n_full = len(enc["case_ids"])
        n_sample = max(1, int(n_full * sample_pct / 100))
        import random
        sampled_ids = random.sample(enc["case_ids"], n_sample)
        sampled_idx = [enc["case_ids"].index(cid) for cid in sampled_ids]
    else:
        sampled_idx = list(range(len(enc["case_ids"])))

    # --- Inference ---
    prog = st.progress(0, text="Đang chạy inference...")
    n_cases = len(sampled_idx)

    try:
        if enc["pipeline_type"] == "prefix":
            # Prefix: enc chứa trực tiếp event_feature_list
            all_preds, all_gt, all_topk_correct = _run_prefix_batch_eval(enc, model, top_k, DEVICE)
        else:
            # GAT: per-case
            all_preds, all_gt, all_topk_correct = [], [], []
            for i, idx in enumerate(sampled_idx):
                prog.progress((i + 1) / n_cases, text=f"Case {i+1}/{n_cases}")
                cid = enc["case_ids"][idx]
                from pipeline import _build_gat_graph_single
                data_list = _build_gat_graph_single(enc, idx, model_type)
                batch = Batch.from_data_list(data_list).to(DEVICE)
                with torch.no_grad():
                    output = model(batch)
                num_events = output.shape[0]
                le_event = enc["le_event"]
                y_encode = enc["y_encode"]
                for step in range(num_events):
                    logits = output[step]
                    pred_idx = int(logits.argmax().item())
                    try:
                        raw = y_encode[idx][step]
                        gt_idx = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
                    except Exception:
                        gt_idx = -1
                    if gt_idx < 0:
                        continue
                    all_preds.append(pred_idx)
                    all_gt.append(gt_idx)
                    top_k_idx = logits.topk(min(top_k, logits.size(0))).indices.tolist()
                    all_topk_correct.append(int(gt_idx in top_k_idx))

        prog.empty()

        st.session_state["be_results"] = {
            "preds": all_preds,
            "gt": all_gt,
            "topk_correct": all_topk_correct,
            "top_k": top_k,
            "model_type": model_type,
            "ds_name": ds_name,
            "meta": meta,
            "le_event": enc["le_event"],
            "sample_pct": sample_pct,
        }

    except Exception as e:
        prog.empty()
        st.error(f"❌ Lỗi inference: {e}")
        st.stop()

# ─── Hiển thị kết quả ─────────────────────────────────────────────────────────
if "be_results" not in st.session_state:
    st.stop()

res = st.session_state["be_results"]
all_preds = res["preds"]
all_gt = res["gt"]
all_topk_correct = res["topk_correct"]
top_k = res["top_k"]
le_event = res["le_event"]
meta = res["meta"]

# ─── Tổng quan metrics ───────────────────────────────────────────────────────
if not all_preds:
    st.warning("Không có prediction nào (dataset trống hoặc tất cả bị mask).")
    st.stop()

top1_acc = np.mean(np.array(all_preds) == np.array(all_gt))
topk_acc = np.mean(all_topk_correct) if all_topk_correct else 0.0
n_preds = len(all_preds)

st.subheader("📈 Metrics tổng quan")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model", res["model_type"].replace("Model", "").replace("Classifier", ""))
c2.metric("Top-1 Accuracy", f"{top1_acc:.2%}")
c3.metric(f"Top-{top_k} Accuracy", f"{topk_acc:.2%}")
c4.metric("Số predictions", f"{n_preds:,}")
c5.metric("Train acc (checkpoint)", f"{meta.get('train_acc', 0):.2%}")
if res["sample_pct"] < 100:
    st.caption(f"⚠️ Chỉ dùng {res['sample_pct']}% dataset")

st.divider()

# ─── Per-class report ─────────────────────────────────────────────────────────
st.subheader("📋 Per-class Classification Report")

try:
    from sklearn.metrics import classification_report
    all_classes = sorted(set(all_gt + all_preds))
    class_names = []
    for c in all_classes:
        try:
            class_names.append(str(le_event.inverse_transform([c])[0]))
        except Exception:
            class_names.append(f"class_{c}")

    report_str = classification_report(
        all_gt, all_preds,
        labels=all_classes,
        target_names=class_names,
        digits=3,
        zero_division=0,
    )
    # Parse sang DataFrame
    lines = [l for l in report_str.split("\n") if l.strip() and "accuracy" not in l.lower()]
    header = ["Class", "Precision", "Recall", "F1-score", "Support"]
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append(parts[:5])
        elif len(parts) == 4:
            rows.append(["(avg)"] + parts[:4])
    if rows:
        df_report = pd.DataFrame(rows, columns=header[:len(rows[0])])
        # Numeric cols
        for col in ["Precision", "Recall", "F1-score"]:
            if col in df_report:
                df_report[col] = pd.to_numeric(df_report[col], errors="coerce")
        st.dataframe(df_report, use_container_width=True, height=min(35 * len(df_report) + 40, 450))
    else:
        st.code(report_str)
except Exception as e:
    st.warning(f"Không thể tạo classification report: {e}")

st.divider()

# ─── Top-K accuracy by class ─────────────────────────────────────────────────
st.subheader("🏆 Accuracy theo class (Top-1)")
from collections import Counter

class_correct = Counter()
class_total = Counter()
for pred, gt in zip(all_preds, all_gt):
    class_total[gt] += 1
    if pred == gt:
        class_correct[gt] += 1

acc_by_class = {}
for c in class_total:
    try:
        label = str(le_event.inverse_transform([c])[0])
    except Exception:
        label = f"class_{c}"
    acc_by_class[label] = class_correct[c] / class_total[c]

df_cls_acc = (
    pd.DataFrame.from_dict(acc_by_class, orient="index", columns=["Accuracy"])
    .sort_values("Accuracy", ascending=False)
)
top20 = df_cls_acc.head(20)
st.bar_chart(top20["Accuracy"])
st.caption("Top 20 class theo accuracy (Top-1). Hover để xem giá trị.")

st.divider()

# ─── Confusion matrix (top events) ──────────────────────────────────────────
with st.expander("🔲 Confusion Matrix (Top-15 classes)", expanded=False):
    try:
        from sklearn.metrics import confusion_matrix
        top_classes = [c for c, _ in Counter(all_gt).most_common(15)]
        mask = [i for i, g in enumerate(all_gt) if g in top_classes]
        if mask:
            gt_sub = [all_gt[i] for i in mask]
            pred_sub = [all_preds[i] for i in mask]
            cm = confusion_matrix(gt_sub, pred_sub, labels=top_classes)
            try:
                labels = [str(le_event.inverse_transform([c])[0]) for c in top_classes]
            except Exception:
                labels = [f"c{c}" for c in top_classes]
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            st.dataframe(df_cm, use_container_width=True)
    except Exception as e:
        st.warning(f"Không thể vẽ confusion matrix: {e}")
