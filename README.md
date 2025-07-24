# 24일차

로보플로우 내에서 30분 정도 모델 train하는 모습<br>
근데 이 방식 안하고도 Download Dataset을 수동으로 눌러서 api가 담긴 코드를 얻었었다.<br>
<img width="958" height="956" alt="image" src="https://github.com/user-attachments/assets/15aa1dcd-4208-4e41-a795-f4528e614a8b" />

## 1. 로보플로우 마무리, 내가 전이학습한 모델을 유튜브 동영상에 적용해보기.
1. 먼저, 첫 실행때 생기는 /content/traffic-detection-1/data.yaml 파일 내부를 수정해야한다.<br>
<img width="337" height="623" alt="image" src="https://github.com/user-attachments/assets/49634712-4b6b-43a7-98c7-2defff01d96e" /><br>
현재 data.yaml의 17~19번째 줄에서는 상위 디렉토리(즉, ../)의 train, valid, test 폴더를 참조하고 있음.<br>
<img width="2560" height="780" alt="image" src="https://github.com/user-attachments/assets/4d22b0a2-270e-4f0a-b106-c51f0b959780" /><br>
이를 Colab 내 실제 폴더 구조인 traffic-detection-1/train/images 경로로 맞춰주고, 셀을 다시 실행한다.<br>
test: ../train/images<br>
train: ../train/images<br>
val: ../train/images<br>
<img width="2037" height="729" alt="image" src="https://github.com/user-attachments/assets/bb60bc9b-dd8e-41fb-865d-f24c6d341a86" /><br>
이제 오류없이 /train/images 경로에 있는 이미지를 토대로 에포크가 실행된다

2. 두번째 실행때 전이학습이 성공하기에, 모델은 두번째 경로인 /content/runs/detect/traffic-custom2/weights/best.pt 에서 다운받으면 된다.
<img width="340" height="474" alt="image" src="https://github.com/user-attachments/assets/07599b04-852a-4a7d-bed7-884cd3691bb7" />

3. 결과영상은 runs/detect/video-result/ 내부에 "temp_video" 라는 이름으로 되어있고, 점선을 많이 라벨링해서 그런지 점선은 잘 인식된다.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0e3bf4bb-950c-4ddd-8a0a-33b116af2412" />

## 2. NVIDIA TAO와 lanenet은 어떤내용이고 코드는 어떻게 되있는지 조사.
NVIDIA TAO Toolkit은 다양한 딥러닝 모델을 사전 학습(pre-trained)된 상태로 제공하고,<br>
전이학습(fine-tuning)을 통해 자신의 데이터에 맞게 재학습할 수 있도록 돕는 도구.<br>

### 자율주행에 활용 가능한 대표적인 모델 목록
| 모델 이름                    | 용도                      | 모델 아키텍처              | 비고             |
| ------------------------ | ----------------------- | -------------------- | -------------- |
| **DetectNet\_v2**        | 객체 탐지 (차량, 보행자 등)       | ResNet, EfficientNet | KITTI 등에서 학습   |
| **YOLOv4 / YOLOv4-Tiny** | 객체 탐지 (실시간)             | CSPDarknet           | 차량 및 도로 객체 인식  |
| **RetinaNet**            | 객체 탐지 (정확도 우선)          | ResNet               | 고정밀 탐지         |
| **UNet / ENet / SegNet** | **도로 차선 및 장면 분할**       | FCN 기반               | 픽셀 단위의 세분화     |
| **CenterPose**           | 보행자 자세 추정 (3D Keypoint) | CenterNet 기반         | 보행자 행동 분석      |
| **ActionRecognitionNet** | 동작 인식 (보행, 정지 등)        | I3D, R2+1D 등         | 비디오 기반         |
| **TrafficCamNet**        | **차량, 보행자, 신호등 탐지**     | DetectNet\_v2 기반     | NVIDIA에서 직접 제공 |
| **DashCamNet**           | 전방 카메라용 객체 탐지           | DetectNet\_v2 기반     | 운전자 보조용        |

### 실제 자율주행에 쓸 수 있는 구조 예
카메라 스트리밍 → YOLO/TrafficCamNet → 물체 인식 결과 처리 → 주행 판단 → RC카 제어 (Jetson Nano/Orin 사용)<br>
또는: UNet → 차선 분할 → steering angle 계산 → 모터 제어<br>

### 모델 중 하나인 TrafficCamNet으로 객체 탐지하는 기본 코드 흐름
TAO Toolkit CLI 또는 Python에서 실행하는 전형적인 학습 워크플로우 예시.

**환경 세팅**
```python
# NGC CLI 로그인
ngc registry model list nvidia/tao/...

# Docker 실행
tao yolo_v4 run bash
```

**spec.yaml 작성 예시**
```python
dataset_config:
  data_sources:
    - label_directory_path: /workspace/data/train/labels
      image_directory_path: /workspace/data/train/images
  validation_data_sources:
    - label_directory_path: /workspace/data/val/labels
      image_directory_path: /workspace/data/val/images
model_config:
  arch: yolo_v4
  nms_iou_threshold: 0.5
  confidence_threshold: 0.5
training_config:
  batch_size_per_gpu: 8
  num_epochs: 80
```

**학습**
```python
tao yolo_v4 train \
  -e /workspace/specs/yolo_v4_train_resnet18_kitti.txt \
  -r /workspace/experiments/yolo_v4/ \
  -k nvidia_tao \
  --gpus 1
```

**추론**
```python
tao yolo_v4 inference \
  -e /workspace/specs/yolo_v4_infer_kitti.txt \
  -o /workspace/results \
  -i /workspace/data/test \
  -k nvidia_tao
```
