"""
loader.py — Load và tái tạo model từ file .pt checkpoint.

Hỗ trợ tất cả 5 model types:
  - DualGATModel               (GATConv.py)
  - DualGATTimeAwareModel      (GATConvTimeDecay.py)
  - DualGAT2EdgesModel         (GATConvStatusEmb.py)
  - DualGATTimeAwareETModel    (GATConvTimeDecayStatusEmb.py)
  - PrefixGCNClassifier        (PrefixEmbeddingGCN.py)
"""

import sys
import os
import torch
from pathlib import Path

# Thêm src/ vào Python path để import các module model
_SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dataset_config import get_model_info


def _infer_model_type(config: dict) -> str | None:
    """
    Tự đoán model_type từ các key có trong config.
    Dùng như fallback khi checkpoint được save trước khi thêm 'model_type'.

    Logic nhận dạng:
      - có 'gcn_hidden_dims'         → PrefixGCNClassifier
      - có 'lambda_decay' + 'num_edge_types' → DualGATTimeAwareETModel
      - có 'lambda_decay'            → DualGATTimeAwareModel
      - có 'num_edge_types'          → DualGAT2EdgesModel
      - còn lại                     → DualGATModel
    """
    keys = set(config.keys())
    if "gcn_hidden_dims" in keys:
        return "PrefixGCNClassifier"
    if "lambda_decay" in keys and "num_edge_types" in keys:
        return "DualGATTimeAwareETModel"
    if "lambda_decay" in keys:
        return "DualGATTimeAwareModel"
    if "num_edge_types" in keys:
        return "DualGAT2EdgesModel"
    return "DualGATModel"


def _build_model(model_type: str, config: dict):
    """
    Tái tạo model object từ model_type và config dict.
    Trả về model ở trạng thái eval, chưa load state_dict.
    """
    info = get_model_info(model_type)
    module_name = info["module"]
    class_name = info["class"]

    # Dynamic import module
    mod = __import__(module_name, fromlist=[class_name])
    ModelClass = getattr(mod, class_name)

    # Xây dựng kwargs từ config (loại bỏ các key không phải constructor param)
    _SKIP_KEYS = {"model_type"}

    if model_type == "DualGATModel":
        model = ModelClass(
            num_event_features=config["num_event_features"],
            num_embedding_features=config["num_embedding_features"],
            embedding_dims=config["embedding_dims"],
            gat_hidden_dim_event=config["gat_hidden_dim_event"],
            gat_hidden_dim_embed=config["gat_hidden_dim_embed"],
            gat_hidden_dim_concat=config["gat_hidden_dim_concat"],
            output_dim=config["output_dim"],
            num_heads=config["num_heads"],
        )

    elif model_type == "DualGATTimeAwareModel":
        model = ModelClass(
            num_event_features=config["num_event_features"],
            num_embedding_features=config["num_embedding_features"],
            embedding_dims=config["embedding_dims"],
            gat_hidden_dim_event=config["gat_hidden_dim_event"],
            gat_hidden_dim_embed=config["gat_hidden_dim_embed"],
            gat_hidden_dim_concat=config["gat_hidden_dim_concat"],
            output_dim=config["output_dim"],
            num_heads=config["num_heads"],
            lambda_decay=config["lambda_decay"],
        )

    elif model_type == "DualGAT2EdgesModel":
        model = ModelClass(
            num_event_features=config["num_event_features"],
            num_embedding_features=config["num_embedding_features"],
            embedding_dims=config["embedding_dims"],
            gat_hidden_dim_event=config["gat_hidden_dim_event"],
            gat_hidden_dim_embed=config["gat_hidden_dim_embed"],
            gat_hidden_dim_concat=config["gat_hidden_dim_concat"],
            output_dim=config["output_dim"],
            num_heads=config["num_heads"],
            num_edge_types=config["num_edge_types"],
            edge_type_dim=config["edge_type_dim"],
        )

    elif model_type == "DualGATTimeAwareETModel":
        model = ModelClass(
            num_event_features=config["num_event_features"],
            num_embedding_features=config["num_embedding_features"],
            embedding_dims=config["embedding_dims"],
            gat_hidden_dim_event=config["gat_hidden_dim_event"],
            gat_hidden_dim_embed=config["gat_hidden_dim_embed"],
            gat_hidden_dim_concat=config["gat_hidden_dim_concat"],
            output_dim=config["output_dim"],
            num_heads=config["num_heads"],
            num_edge_types=config["num_edge_types"],
            edge_type_dim=config["edge_type_dim"],
            lambda_decay=config["lambda_decay"],
        )

    elif model_type == "PrefixGCNClassifier":
        model = ModelClass(
            num_event_features=config["num_event_features"],
            gcn_hidden_dims=config["gcn_hidden_dims"],
            num_embedding_features=config["num_embedding_features"],
            embedding_dims=config["embedding_dims"],
            gcn_hidden_dims_embedding=config["gcn_hidden_dims_embedding"],
            gcn_hidden_dims_concat=config["gcn_hidden_dims_concat"],
            num_sequence_features=config["num_sequence_features"],
            fc_hidden_dims=config["fc_hidden_dims"],
            fc_hidden_dims_concat=config["fc_hidden_dims_concat"],
            output_dim=config["output_dim"],
        )

    else:
        raise ValueError(f"model_type '{model_type}' chưa được hỗ trợ trong _build_model()")

    return model


def load_checkpoint(pt_path: str, device: torch.device = None):
    """
    Load file .pt và trả về:
      - model      : model đã rebuild + load state_dict, ở trạng thái eval()
      - config     : dict config đã lưu trong checkpoint
      - checkpoint : toàn bộ dict checkpoint (để lấy attention_data, metrics, ...)

    Tham số:
      pt_path : đường dẫn tới file .pt
      device  : torch.device để load model lên (mặc định: CPU)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Không tìm thấy file model: '{pt_path}'")

    checkpoint = torch.load(pt_path, map_location=device)

    config = checkpoint.get("config")
    if config is None:
        raise KeyError(
            f"Checkpoint '{pt_path}' không có key 'config'. "
            "Vui lòng re-train và save lại model."
        )

    model_type = config.get("model_type")
    if model_type is None:
        model_type = _infer_model_type(config)
        if model_type is None:
            raise KeyError(
                f"Không thể xác định model_type từ config trong '{pt_path}'. "
                "Vui lòng re-train model để lưu 'model_type' vào config."
            )

    model = _build_model(model_type, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config, checkpoint


def get_model_metadata(checkpoint: dict) -> dict:
    """
    Trích xuất thông tin hiển thị từ checkpoint.
    Trả về dict với các key: model_type, epoch, train_acc, test_acc, train_loss, test_loss,
                              has_attention, output_dim
    """
    config = checkpoint.get("config", {})
    model_type = config.get("model_type") or _infer_model_type(config) or "Unknown"
    try:
        info = get_model_info(model_type)
    except ValueError:
        info = {"has_attention": False, "edge_type": "unknown"}
    return {
        "model_type": model_type,
        "epoch": checkpoint.get("epoch", "N/A"),
        "train_acc": checkpoint.get("train_acc"),
        "test_acc": checkpoint.get("test_acc"),
        "train_loss": checkpoint.get("train_loss"),
        "test_loss": checkpoint.get("test_loss"),
        "has_attention": info.get("has_attention", False),
        "edge_type": info.get("edge_type", "unknown"),
        "output_dim": config.get("output_dim"),
    }


_SCAN_CACHE: dict[str, list[dict]] = {}


def scan_model_dir(models_dir: str, force_refresh: bool = False) -> list[dict]:
    """
    Quét thư mục và trả về danh sách thông tin của tất cả file .pt.
    Mỗi entry: { 'filename', 'path', 'model_type', 'test_acc', 'epoch' }

    Kết quả được cache trong memory để tránh torch.load() mỗi lần sidebar render.
    Dùng force_refresh=True nếu cần làm mới (e.g. sau khi thêm file .pt mới).
    """
    if not force_refresh and models_dir in _SCAN_CACHE:
        return _SCAN_CACHE[models_dir]

    results = []
    models_path = Path(models_dir)
    if not models_path.is_dir():
        return results

    for pt_file in sorted(models_path.glob("*.pt")):
        entry = {
            "filename": pt_file.name,
            "path": str(pt_file),
            "model_type": None,
            "test_acc": 0.0,
            "epoch": 0,
        }
        try:
            ckpt = torch.load(str(pt_file), map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            mt = cfg.get("model_type") or _infer_model_type(cfg)
            entry["model_type"] = mt
            entry["test_acc"] = float(ckpt.get("test_acc") or 0.0)
            entry["epoch"] = int(ckpt.get("epoch") or 0)
        except Exception:
            pass
        results.append(entry)

    _SCAN_CACHE[models_dir] = results
    return results
