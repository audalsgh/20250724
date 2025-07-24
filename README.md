# 24일차

## 로보플로우 마무리, 내가 전이학습한 모델을 유튜브 동영상에 적용해보기.
1. 먼저, 첫 실행때 생기는 /content/traffic-detection-1/data.yaml 파일 내부를 수정해야한다.
<img width="337" height="623" alt="image" src="https://github.com/user-attachments/assets/49634712-4b6b-43a7-98c7-2defff01d96e" /><br>
현재 data.yaml의 17~19번째 줄에서는 상위 디렉토리(즉, ../)의 train, valid, test 폴더를 참조하고 있음.<br>

<img width="2560" height="780" alt="image" src="https://github.com/user-attachments/assets/4d22b0a2-270e-4f0a-b106-c51f0b959780" /><br>
이를 Colab 내 실제 폴더 구조인 traffic-detection-1/train/images 경로로 맞춰주고, 셀을 다시 실행한다.<br>
test: ../train/images<br>
train: ../train/images<br>
val: ../train/images<br>

<img width="2037" height="729" alt="image" src="https://github.com/user-attachments/assets/bb60bc9b-dd8e-41fb-865d-f24c6d341a86" /><br>
이제 오류없이 에포크가 실행된다

2. 두번째 실행때 전이학습이 성공하기에, 모델은 두번째 경로인 /content/runs/detect/traffic-custom2/weights/best.pt 에서 다운받으면 된다.
<img width="340" height="474" alt="image" src="https://github.com/user-attachments/assets/07599b04-852a-4a7d-bed7-884cd3691bb7" />

3. 결과영상은 runs/detect/video-result/ 내부에 "temp_video" 라는 이름으로 되어있고, 점선을 많이 라벨링해서 그런지 점선은 잘 인식된다.
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0e3bf4bb-950c-4ddd-8a0a-33b116af2412" />

