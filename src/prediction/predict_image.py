# -*- coding: utf-8 -*-
"""
Predict one traffic sign image from a user-provided path.

Run from project root:
    python src\prediction\predict_image.py

The script will ask:
    请输入图片地址：
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_model import build_cnn_model  # noqa: E402


IMG_SIZE = 64
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "resnet18_fpn_best.pth",
    PROJECT_ROOT / "models" / "resnet18_fpn_final.pth",
]
MAPPING_PATH = PROJECT_ROOT / "models" / "label_mapping.json"


GTSRB_CLASS_NAMES = {
    0: "Speed limit 20 km/h / 限速20",
    1: "Speed limit 30 km/h / 限速30",
    2: "Speed limit 50 km/h / 限速50",
    3: "Speed limit 60 km/h / 限速60",
    4: "Speed limit 70 km/h / 限速70",
    5: "Speed limit 80 km/h / 限速80",
    6: "End of speed limit 80 km/h / 解除限速80",
    7: "Speed limit 100 km/h / 限速100",
    8: "Speed limit 120 km/h / 限速120",
    9: "No passing / 禁止超车",
    10: "No passing for vehicles over 3.5 tons / 大型车辆禁止超车",
    11: "Right-of-way at next intersection / 路口优先通行",
    12: "Priority road / 优先道路",
    13: "Yield / 让行",
    14: "Stop / 停车让行",
    15: "No vehicles / 禁止车辆通行",
    16: "Vehicles over 3.5 tons prohibited / 禁止大型车辆通行",
    17: "No entry / 禁止驶入",
    18: "General caution / 注意危险",
    19: "Dangerous curve left / 左急转弯",
    20: "Dangerous curve right / 右急转弯",
    21: "Double curve / 连续弯路",
    22: "Bumpy road / 颠簸路面",
    23: "Slippery road / 易滑路面",
    24: "Road narrows on the right / 右侧道路变窄",
    25: "Road work / 道路施工",
    26: "Traffic signals / 交通信号灯",
    27: "Pedestrians / 注意行人",
    28: "Children crossing / 注意儿童",
    29: "Bicycles crossing / 注意自行车",
    30: "Beware of ice/snow / 注意冰雪",
    31: "Wild animals crossing / 注意野生动物",
    32: "End of all restrictions / 解除所有限制",
    33: "Turn right ahead / 前方右转",
    34: "Turn left ahead / 前方左转",
    35: "Ahead only / 直行",
    36: "Go straight or right / 直行或右转",
    37: "Go straight or left / 直行或左转",
    38: "Keep right / 靠右行驶",
    39: "Keep left / 靠左行驶",
    40: "Roundabout mandatory / 环岛行驶",
    41: "End of no passing / 解除禁止超车",
    42: "End of no passing over 3.5 tons / 解除大型车辆禁止超车",
}


def find_model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"未找到模型文件，请先训练模型。尝试查找：\n{candidates}")


def load_label_mapping(checkpoint: dict) -> tuple[int, dict[int, int]]:
    if "index_to_class" in checkpoint:
        index_to_class = {int(k): int(v) for k, v in checkpoint["index_to_class"].items()}
        num_classes = int(checkpoint.get("num_classes", len(index_to_class)))
        return num_classes, index_to_class

    if MAPPING_PATH.exists():
        payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        index_to_class = {int(k): int(v) for k, v in payload["index_to_class"].items()}
        num_classes = int(payload.get("num_classes", len(index_to_class)))
        return num_classes, index_to_class

    return 43, {i: i for i in range(43)}


def load_model(device: torch.device):
    model_path = find_model_path()
    checkpoint = torch.load(model_path, map_location=device)

    num_classes, index_to_class = load_label_mapping(checkpoint)
    model = build_cnn_model(num_classes=num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"已加载模型：{model_path}")
    return model, index_to_class


def clean_user_path(raw_path: str) -> Path:
    raw_path = raw_path.strip().strip('"').strip("'")
    return Path(raw_path)


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"无法读取图片，请检查文件格式：{image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0)
    return tensor, image_rgb


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, index_to_class: dict[int, int], device: torch.device):
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_index = int(torch.argmax(probs).item())
    pred_class_id = int(index_to_class[pred_index])
    confidence = float(probs[pred_index].item())

    topk = min(5, probs.numel())
    top_probs, top_indices = torch.topk(probs, k=topk)
    top_results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        class_id = int(index_to_class[int(idx)])
        top_results.append((class_id, float(prob)))

    return pred_class_id, confidence, top_results


def show_image(image_rgb: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备：{device}")

    model, index_to_class = load_model(device)

    raw_path = input("请输入图片地址：")
    image_path = clean_user_path(raw_path)

    image_tensor, original_rgb = preprocess_image(image_path)
    pred_class_id, confidence, top_results = predict(model, image_tensor, index_to_class, device)
    class_name = GTSRB_CLASS_NAMES.get(pred_class_id, "未知类别")

    print("\n预测结果：")
    print(f"图片地址：{image_path}")
    print(f"预测类别ID：{pred_class_id}")
    print(f"预测类别名称：{class_name}")
    print(f"置信度：{confidence:.4f}")

    print("\nTop-5 预测：")
    for rank, (class_id, prob) in enumerate(top_results, start=1):
        name = GTSRB_CLASS_NAMES.get(class_id, "未知类别")
        print(f"{rank}. ClassId={class_id}, prob={prob:.4f}, name={name}")

    show_image(original_rgb, f"Pred: ClassId={pred_class_id}, Conf={confidence:.3f}")


if __name__ == "__main__":
    main()