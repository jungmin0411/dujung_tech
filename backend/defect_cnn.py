"""용접 불량 종류 분류용 CNN — 남우현님이 실제로 학습 완료한 커스텀 8-class CNN
(2·3단계 파이프라인 중 3단계 담당, custom_cnn_8class_best.pt).

아키텍처가 학습 시점과 100% 동일해야 state_dict를 그대로 불러올 수 있으므로
레이어 구성/순서를 임의로 바꾸면 안 됨. 전처리(64x64 리사이즈 + ToTensor만, 정규화 없음)도
학습 때와 동일하게 맞춰야 함 — cnn_how_to_use_manual.ipynb 기준.

※ 팀원 지침(중요): YOLO와 CNN은 같은 입력 이미지를 "병렬"로 받는 구조.
YOLO가 찾은 박스로 재crop해서 CNN에 넣으면 안 됨 — 각자 독립적으로 원본(크롭 안 된) 이미지를
그대로 입력받는다.
"""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 고정 순서 (학습 시점 그대로) — 절대 순서를 바꾸면 안 됨
CLASS_NAMES_EN = [
    "Good Weld", "Porosity", "Spatter", "Burn-through",
    "Overlap", "Undercut", "Crack", "Lack of Fusion",
]
CLASS_NAMES_KR = [
    "정상", "기공", "스패터", "화상(Burn-through)",
    "오버랩", "언더컷", "균열", "융합불량",
]
INPUT_SIZE = 64

_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
])


class SimpleWeldCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES_EN)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_model(num_classes: int = len(CLASS_NAMES_EN)) -> nn.Module:
    return SimpleWeldCNN(num_classes=num_classes)


def load_model(weights_path: str, device: str = "cpu") -> nn.Module:
    model = build_model()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_defect_type(model: nn.Module, image: Image.Image, exclude_good: bool = True) -> str:
    """image는 YOLO 박스로 자른 crop이 아니라 원본 입력 이미지 그대로 넣는다 (팀원 지침).
    exclude_good=True면 "정상" 클래스는 후보에서 제외하고 나머지 7개 불량 종류 중 최댓값을
    고른다 — 이 함수는 YOLO가 이미 "bad"로 판정했을 때만 호출되기 때문."""
    tensor = _transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        if exclude_good:
            logits = logits.clone()
            logits[0, 0] = float("-inf")
        idx = int(torch.argmax(logits, dim=1)[0])
    return CLASS_NAMES_KR[idx]
