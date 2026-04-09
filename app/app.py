"""
app.py — Streamlit entry point cho TemporalAwareGNNs demo app.

Chạy từ thư mục app/:
    cd app && streamlit run app.py

Hoặc từ project root:
    streamlit run app/app.py
"""

import sys
import os
from pathlib import Path

# ── Suppress TF/XLA/protobuf chargement avant tout import ────────────────────
# TensorFlow và PyTorch cùng tồn tại trong env → protobuf descriptor conflict
# khi TF loads XLA ở thread phụ (Streamlit page runner).
# Fix: set env vars + force import thứ tự đúng ngay ở main thread.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")       # tắt TF verbose logging
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")         # tắt CUDA cho TF
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")               # tránh GUI toolkit conflict
os.environ.setdefault("OMP_NUM_THREADS", "4")             # giới hạn OMP threads

# Warmup imports — phải load TRƯỚC khi Streamlit khởi động page runner threads
# để protobuf descriptors được registered ở main thread (tránh race condition)
try:
    import tensorflow as _tf  # noqa: F401 — import TF trước để init protobuf của TF
    _tf_version = _tf.__version__
except ImportError:
    _tf_version = None

import torch as _torch  # noqa: F401 — import torch sau để avoid XLA / protobuf conflict
import torch_geometric  # noqa: F401

# Đảm bảo app/core và src/ luôn được tìm thấy
_APP_DIR = Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR.parent / "src"
for d in (_APP_DIR / "core", _SRC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

# Pre-import all model modules in main thread (Streamlit re-uses same sys.modules)
import warnings
warnings.filterwarnings("ignore")
try:
    import GATConv, GATConvTimeDecay, GATConvStatusEmb  # noqa: F401
    import GATConvTimeDecayStatusEmb, PrefixEmbeddingGCN  # noqa: F401
except Exception:
    pass  # Lỗi import model sẽ được báo khi mở trang tương ứng

import streamlit as st

st.set_page_config(
    page_title="Temporal-Aware GNN — Next Event Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔮 Temporal-Aware GNN — Next Event Prediction")
st.markdown(
    """
    Ứng dụng demo cho bộ mô hình **Temporal-Aware Graph Neural Network** 
    dự đoán sự kiện tiếp theo trong quy trình nghiệp vụ (Business Process Management).

    ---
    ### Điều hướng

    Sử dụng menu bên trái để chuyển giữa các trang:

    | Trang | Mô tả |
    |-------|-------|
    | 🎯 **Demo Case** | Chọn một case, xem dự đoán per-step với xác suất top-K |
    | 📊 **Batch Evaluation** | Đánh giá toàn bộ dataset: accuracy, top-K, per-class report |
    | 🧠 **Attention Analysis** | Trực quan hóa attention/decay weights của Time-Aware models |

    ---
    ### Các model hỗ trợ

    | Model | Kiến trúc | Attention |
    |-------|-----------|-----------|
    | `DualGATModel` | Dual-path GAT + time edge | — |
    | `DualGATTimeAwareModel` | Dual-path GAT + time-decay attention | ✅ |
    | `DualGAT2EdgesModel` | Dual-path GAT + transition type embedding | — |
    | `DualGATTimeAwareETModel` | Dual-path GAT + decay + transition | ✅ |
    | `PrefixGCNClassifier` | GCN + prefix window + sequence features | — |

    ---
    ### Bắt đầu nhanh

    1. Mở trang **Demo Case** từ menu bên trái  
    2. Chọn file model `.pt` từ thư mục `output/models/`  
    3. Chọn dataset tương ứng  
    4. Nhấn **Load & Encode** rồi chọn một case để xem dự đoán
    """
)

st.info(
    "📁 File model được đọc từ `output/models/`. "
    "Đảm bảo đã train ít nhất một model trước khi sử dụng.",
    icon="ℹ️",
)
