"""용접 불량 세부유형(균열/기공/언더컷 등) 분류용 CNN.

지금은 미학습 상태(랜덤 초기화 가중치, defect_cnn_weights.pt)로 자리만 잡아둔 것 —
팀원이 실제 학습을 마치면 같은 클래스 순서·같은 입력 크기로 저장한
defect_cnn_weights.pt 파일 하나만 이 이름 그대로 덮어쓰면 코드 수정 없이 바로 적용됨.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Dashboard.jsx의 DEFECT_TYPES와 순서를 반드시 맞춰야 함 (인덱스로 매칭되기 때문)
DEFECT_CLASSES = ["균열", "기공", "언더컷", "융합불량", "용입부족", "오버랩", "스패터"]
INPUT_SIZE = 64  # 정사각형으로 리사이즈해서 넣음


class DefectCNN(nn.Module):
    def __init__(self, num_classes: int = len(DEFECT_CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 64 -> 32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(weights_path: str, device: str = "cpu") -> DefectCNN:
    model = DefectCNN()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _preprocess(crop_img: Image.Image) -> torch.Tensor:
    crop_img = crop_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(crop_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def predict_defect_type(model: DefectCNN, crop_img: Image.Image) -> str:
    tensor = _preprocess(crop_img)
    with torch.no_grad():
        logits = model(tensor)
        idx = int(torch.argmax(logits, dim=1)[0])
    return DEFECT_CLASSES[idx]
