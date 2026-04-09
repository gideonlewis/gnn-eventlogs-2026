"""
Cấu hình per-dataset: ánh xạ từng dataset sang các tham số encoding tương ứng.
Được dùng bởi pipeline.py để tái tạo quá trình tiền xử lý từ các notebook.
"""

# Đường dẫn gốc tương đối từ thư mục app/
OUTPUT_DIR = "../output"
MODELS_DIR = "../output/models"

# ─── Cấu hình dataset ─────────────────────────────────────────────────────────
# Mỗi entry gồm:
#   file         : đường dẫn CSV (tương đối từ thư mục app/)
#   display_name : tên hiển thị trên UI
#   core_event   : cột chứa tên sự kiện (dùng làm input/output chính)
#   case_index   : cột định danh case/sequence
#   time_col     : cột timestamp
#   cat_col_event: cột categorical ở cấp event (được one-hot encode)
#   num_col_event: cột numerical ở cấp event (được min-max scale)
#   cat_col_seq  : cột categorical ở cấp sequence (sequence-level feature)
#   num_col_seq  : cột numerical ở cấp sequence
#   seq_cols     : cột cần giữ khi tạo sequence DataFrame (luôn bao gồm case_index)
#   display_cols : cột hiển thị trên bảng lịch sử sự kiện trong UI
#   ec1_as_str   : True nếu cột ec1 cần ép sang str trước khi encode (BPI12/BPI12w)
#   min_seq_size : số sự kiện tối thiểu để giữ 1 case (lọc noise)

DATASET_CONFIGS = {
    "Helpdesk": {
        "file": f"{OUTPUT_DIR}/helpdesk.csv",
        "display_name": "Helpdesk",
        "core_event": "event",
        "case_index": "sequence",
        "time_col": "time",
        "cat_col_event": ["ec1"],
        "num_col_event": [],
        "cat_col_seq": ["sc1"],
        "num_col_seq": [],
        "seq_cols": ["sequence", "sc1"],
        "display_cols": ["event", "time", "ec1", "sc1"],
        "ec1_as_str": False,
        "min_seq_size": 2,
    },
    "BPI12w": {
        "file": f"{OUTPUT_DIR}/BPI12w.csv",
        "display_name": "BPI Challenge 2012 (W)",
        "core_event": "event",
        "case_index": "sequence",
        "time_col": "time",
        "cat_col_event": ["ec1"],
        "num_col_event": [],
        "cat_col_seq": [],
        "num_col_seq": ["sn1"],
        "seq_cols": ["sequence", "sn1"],
        "display_cols": ["event", "event_label", "status", "time", "ec1", "sn1"],
        "ec1_as_str": True,
        "min_seq_size": 2,
    },
    "BPI12": {
        "file": f"{OUTPUT_DIR}/BPI12.csv",
        "display_name": "BPI Challenge 2012",
        "core_event": "event",
        "case_index": "sequence",
        "time_col": "time",
        "cat_col_event": ["ec1"],
        "num_col_event": [],
        "cat_col_seq": [],
        "num_col_seq": ["sn1"],
        "seq_cols": ["sequence", "sn1"],
        "display_cols": ["event", "event_label", "status", "time", "ec1", "sn1"],
        "ec1_as_str": True,
        "min_seq_size": 2,
    },
    "BPI13i": {
        "file": f"{OUTPUT_DIR}/BPI13i.csv",
        "display_name": "BPI Challenge 2013 (Incidents)",
        "core_event": "event",
        "case_index": "sequence",
        "time_col": "time",
        "cat_col_event": ["ec1", "ec4"],
        "num_col_event": [],
        "cat_col_seq": ["sc1", "sc2", "sc3"],
        "num_col_seq": [],
        "seq_cols": ["sequence", "sc1", "sc2", "sc3"],
        "display_cols": ["event", "event_label", "status", "time", "ec1", "ec4", "sc1"],
        "ec1_as_str": False,
        "min_seq_size": 2,
    },
    "BPI13c": {
        "file": f"{OUTPUT_DIR}/BPI13c.csv",
        "display_name": "BPI Challenge 2013 (Closed)",
        "core_event": "event",
        "case_index": "sequence",
        "time_col": "time",
        "cat_col_event": ["ec1", "ec4"],
        "num_col_event": [],
        "cat_col_seq": ["sc1", "sc2", "sc3"],
        "num_col_seq": [],
        "seq_cols": ["sequence", "sc1", "sc2", "sc3"],
        "display_cols": ["event", "event_label", "status", "time", "ec1", "ec4", "sc1"],
        "ec1_as_str": False,
        "min_seq_size": 2,
    },
}

# ─── Ánh xạ model_type → module Python ────────────────────────────────────────
# Dùng bởi loader.py để import đúng class khi rebuild model từ checkpoint.

MODEL_REGISTRY = {
    "DualGATModel": {
        "module": "GATConv",
        "class": "DualGATModel",
        "has_attention": False,
        "edge_type": "time",          # chỉ dùng time diff làm edge attr
    },
    "DualGATTimeAwareModel": {
        "module": "GATConvTimeDecay",
        "class": "DualGATTimeAwareModel",
        "has_attention": True,         # lưu attention_data trong checkpoint
        "edge_type": "time_decay",
    },
    "DualGAT2EdgesModel": {
        "module": "GATConvStatusEmb",
        "class": "DualGAT2EdgesModel",
        "has_attention": False,
        "edge_type": "time_and_transition",  # time diff + transition type embedding
    },
    "DualGATTimeAwareETModel": {
        "module": "GATConvTimeDecayStatusEmb",
        "class": "DualGATTimeAwareETModel",
        "has_attention": True,
        "edge_type": "time_decay_and_transition",
    },
    "PrefixGCNClassifier": {
        "module": "PrefixEmbeddingGCN",
        "class": "PrefixGCNClassifier",
        "has_attention": False,
        "edge_type": "none",           # GCN thuần, không dùng edge attr
    },
}

# ─── Helper functions ──────────────────────────────────────────────────────────

def get_dataset_config(name: str) -> dict:
    """Trả về config của dataset theo tên. Raise ValueError nếu không tìm thấy."""
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{name}' không tồn tại. "
            f"Các dataset hỗ trợ: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[name]


def get_model_info(model_type: str) -> dict:
    """Trả về thông tin model theo model_type. Raise ValueError nếu không tìm thấy."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"model_type '{model_type}' chưa được đăng ký. "
            f"Các model hỗ trợ: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def list_datasets() -> list[str]:
    """Trả về danh sách tên tất cả datasets."""
    return list(DATASET_CONFIGS.keys())


def list_models() -> list[str]:
    """Trả về danh sách tất cả model_type đã đăng ký."""
    return list(MODEL_REGISTRY.keys())
