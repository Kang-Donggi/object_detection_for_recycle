# object detection for recycle

### 개요:
- 쓰레기 촬영 데이터에서 object detection 수행

### 데이터:
- 9754장의 쓰레기 촬영 데이터 <br/>
![화면 캡처 2024-02-25 005358](https://github.com/KANG-dg/object_detection_for_recycle/assets/121837927/b2d926cf-7ec1-4c06-a949-8ff566057eb5)

- 10개의 class 존재 <br/>
  [General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, clothing]
  
### 주요 사용기법:
- EDA 시행 결과 데이터 불균형, bounding box 크기 차이 확인 <br/>
![imbalance](https://github.com/KANG-dg/object_detection_for_recycle/assets/121837927/76592624-d7ad-4003-9dfa-f0b08da30b1c)
- 데이터 불균형 완화를 위해 데이터가 부족한 class에 oversampling 진행 <br/>
![oversampling](https://github.com/KANG-dg/object_detection_for_recycle/assets/121837927/00dd5045-2a72-4cc1-8366-9583778c8328)
- 모델 학습 결과를 confusion matrix로 확인했을 때 특정 객체를 잘 탐지하는 모델 확인 <br/>
![confusion](https://github.com/KANG-dg/object_detection_for_recycle/assets/121837927/cdf1107f-7a93-403c-ad09-4cb5ed35e129)
- Faster-RCNN, YOLOv8, RT-DETR 세가지 모델 앙상블

### 결과:
- oversampling은 유의미한 효과 다만 일정 이상 사용 시 overfitting 발생
- 다른 특성을 가지는 모델을 앙상블 하는 것은 test set에 대한 mAP score를 높이는 것에는 효과가 있었음

### 회고
- 앙상블을 통해 여러 개의 bounding box를 그리는 방법은 map를 높이는 것에는 유의미하지만 <br/>
  실제 산업에서는 많은 컴퓨팅 자원이 소모됨과 동시에 판단에 혼동을 줄 수 있을 것으로 생각됨
![ensemble](https://github.com/KANG-dg/object_detection_for_recycle/assets/121837927/70b7095f-f1ed-4ca1-9490-d61af3892b01)
