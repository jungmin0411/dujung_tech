"""용접 불량 세부유형(균열/기공/언더컷 등) 분류용 CNN.

팀원의 실제 학습 코드와 구조를 맞춰서 만듦 — torchvision ResNet18(ImageNet 사전학습)
백본 + 마지막 FC만 7-클래스로 교체, 입력 224x224. 지금은 미학습(ImageNet 가중치 +
랜덤 초기화된 FC) 상태 — 팀원이 학습 완료하면 defect_cnn_weights.pt만 같은 이름으로
덮어쓰면 코드 수정 없이 바로 적용됨.

※ 클래스 순서 관련 중요 주의사항 ※
팀원 쪽 학습 코드가 지금 `sorted(set(labels))`로 클래스를 "알파벳순 자동 정렬"하고 있어서,
아래 DEFECT_CLASSES 순서는 팀원이 보고해준 라벨 문자열(Crack/Porosity/Undercut/
Lack of Fusion/Lack of Penetration/Overlap/Spatter)의 알파벳순을 그대로 옮겨 적어둔 것.
팀원이 라벨 문자열을 하나라도 바꾸거나 데이터셋 라벨 집합이 바뀌면 이 순서가 소리 없이
달라질 수 있음 — 팀원 쪽에서 고정 리스트(하드코딩)로 바꾸는 게 안전함. 그 전까지는
학습 완료 시점에 팀원한테 최종 클래스 순서를 다시 한번 확인받아야 함.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

# 알파벳순(영문 라벨 기준) 정렬 결과를 그대로 옮긴 것 — 팀원 라벨 문자열이 바뀌면 같이 바뀌어야 함
DEFECT_CLASSES = ["균열", "융합불량", "용입부족", "오버랩", "기공", "스패터", "언더컷"]
# Crack        Lack of Fusion  Lack of Penetration  Overlap  Porosity  Spatter  Undercut
INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int = len(DEFECT_CLASSES)) -> nn.Module:
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path: str, device: str = "cpu") -> nn.Module:
    model = build_model()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _preprocess(crop_img: Image.Image) -> torch.Tensor:
    crop_img = crop_img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.asarray(crop_img, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)


def predict_defect_type(model: nn.Module, crop_img: Image.Image) -> str:
    tensor = _preprocess(crop_img)
    with torch.no_grad():
        logits = model(tensor)
        idx = int(torch.argmax(logits, dim=1)[0])
    return DEFECT_CLASSES[idx]
