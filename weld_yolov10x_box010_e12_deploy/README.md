# Weld YOLOv10x Base-soft Box 0.10 E12

용접 부위를 `good`, `bad` 두 클래스로 검출하는 YOLOv10x 모델 패키지입니다.

## 모델

- 모델: YOLOv10x
- 선택 체크포인트: Base-soft + `box=0.10`, human epoch E12
- 입력 크기: 640
- 기본 confidence: 0.60
- 클래스: `0=good`, `1=bad`
- 학습 데이터: 고정 train 1,606장 / val 402장
- DU22 2.0x 결과: TP 7, FP 0, FN 5, precision 1.0, recall 0.583

`conf=0.60`은 제공된 DU22 2.0x 검사셋에서 선택한 값입니다. 원본 전체
화면에서는 0.725가 검사셋 기준값이므로 촬영 거리와 환경에 맞춰 조정하십시오.

## 설치

Python 3.10 이상을 권장합니다. NVIDIA GPU를 쓸 경우 환경에 맞는 PyTorch를
먼저 설치한 후 다음 명령을 실행합니다.

```bash
python3 -m pip install -r requirements.txt
```

## 실행

단일 이미지:

```bash
python3 predict_weld.py /path/to/image.jpg
```

이미지 폴더:

```bash
python3 predict_weld.py /path/to/images --conf 0.60 --name weld_result
```

CPU 실행:

```bash
python3 predict_weld.py /path/to/images --device cpu
```

예측 라벨과 confidence도 저장:

```bash
python3 predict_weld.py /path/to/images --save-txt --save-conf
```

결과는 기본적으로 패키지의 `runs/prediction` 아래에 저장됩니다.

## 파일

- `model/weld_yolov10x_box010_e12.pt`: 추론용 가중치
- `predict_weld.py`: 이미지/폴더/영상 추론 코드
- `example_result/`: DU22 2.0x 결과 시트
- `metadata/`: 검증 보고서, 학습 인자, 출처와 SHA-256

색상은 Ultralytics 기본 출력 팔레트를 사용합니다. `example_result` 시트에서는
청록색이 GT, 초록색이 good 예측, 빨간색이 bad 예측입니다.
