# 7.2 표정과 제스처 기반의 감성 인식 시스템

## 개요

감성 인식 시스템에서 표정과 제스처는 중요한 비언어적 정보를 제공합니다. 이 장에서는 오픈소스 기반의 표정 정보 처리 및 추출 방법과 제스처 기반 감성 인식 시스템에 대해 학습합니다.

## 1. 오픈소스 기반의 표정 정보 처리 및 추출

### 얼굴 표정 기반 감성 인식 시스템 개발 목표

- **핵심 아이디어**: 감성 표정을 통해 움직이는 근육들의 정보를 인식
- **기술적 접근**: 영상 처리를 통한 얼굴 근육 움직임 분석
- **활용 분야**: 
  - 사용자 경험(UX) 개선
  - 인간-컴퓨터 상호작용 향상
  - 감성 컴퓨팅 응용

### FACS (Facial Action Coding System)

#### 시스템 개요

- **개발자**: 폴 에크만(Paul Ekman)과 월러스 프리젠(Wallace Friesen)
- **개발 연도**: 1978년
- **기본 원리**: 사람의 얼굴 근육을 해부학적 기반으로 분석

#### 핵심 개념: Action Unit (AU)

- **정의**: 각 얼굴 근육의 특정 움직임을 나타내는 기본 단위
- **범위**: AU 1번부터 AU 68번까지 다양한 근육 정보 정의
- **특징**: 
  - 해부학적으로 독립적인 근육 움직임
  - 객관적이고 재현 가능한 분석 기준
  - 문화적 편견을 배제한 과학적 접근

#### 주요 Action Unit 분류

**상부 얼굴 (Upper Face)**:
- AU1: Inner Brow Raise (내측 눈썹 올리기)
- AU2: Outer Brow Raise (외측 눈썹 올리기)
- AU4: Brow Lowerer (눈썹 내리기)
- AU5: Upper Lid Raise (위 눈꺼풀 올리기)
- AU6: Cheek Raise (볼 올리기)
- AU7: Lid Tighten (눈꺼풀 조이기)

**하부 얼굴 (Lower Face)**:
- AU9: Nose Wrinkle (코 주름)
- AU10: Upper Lip Raise (윗입술 올리기)
- AU12: Lip Corner Puller (입꼬리 당기기)
- AU15: Lip Corner Depressor (입꼬리 내리기)
- AU17: Chin Raise (턱 올리기)
- AU18: Lip Pucker (입술 오므리기)

### 감성별 AU 조합 패턴

#### 행복한 표정 (Happiness)
- **주요 AU**: AU6 + AU12
- **특징**:
  - AU6: 볼이 올라가며 눈가 주름 생성
  - AU12: 입꼬리가 올라감
  - 듀셴 스마일(Duchenne Smile): 진정한 행복의 표현

#### 슬픈 표정 (Sadness)
- **주요 AU**: AU1 + AU4 + AU15
- **특징**:
  - AU1: 눈썹 안쪽이 올라감
  - AU4: 눈썹이 내려가며 찡그림
  - AU15: 입꼐리가 내려감

#### 놀람 표정 (Surprise)
- **주요 AU**: AU1 + AU2 + AU5 + AU26
- **특징**:
  - AU1+AU2: 눈썹이 올라감
  - AU5: 위 눈꺼풀이 올라감
  - AU26: 입이 벌어짐

#### 화남 표정 (Anger)
- **주요 AU**: AU4 + AU5 + AU7 + AU23
- **특징**:
  - AU4: 눈썹이 내려감
  - AU5+AU7: 눈이 날카로워짐
  - AU23: 입술이 조여짐

## 2. OpenCV 기반 얼굴 표정 인식 구현

### Landmark 기반 접근법

#### Facial Landmark 개념

- **정의**: 얼굴의 특징적인 점들의 좌표 정보
- **총 개수**: 일반적으로 68개 포인트 사용
- **활용**: AU 움직임 정보 추출의 기반 데이터

#### Landmark 좌표 인덱스 구조

**얼굴 윤곽 (Face Contour)**:
- 포인트 0-16: 턱선을 따라 좌측에서 우측으로

**눈썹 (Eyebrows)**:
- 우측 눈썹: 포인트 17-21
- 좌측 눈썹: 포인트 22-26

**눈 (Eyes)**:
- 우측 눈: 포인트 36-41
- 좌측 눈: 포인트 42-47

**코 (Nose)**:
- 콧대: 포인트 27-30
- 콧구멍: 포인트 31-35

**입 (Mouth)**:
- 외측 윤곽: 포인트 48-59
- 내측 윤곽: 포인트 60-67

### AU와 Landmark 매핑 테이블

| AU 번호 | AU 이름 | FACS 한국어명 | 해당 Landmark 인덱스 |
|---------|---------|---------------|---------------------|
| AU1 | Inner Brow Raise | 눈썹 안쪽이 올라감 | 좌: 17,18,19,20,21<br>우: 22,23,24,25,26 |
| AU2 | Outer Brow Raise | 눈썹 바깥쪽이 올라감 | 17,21,22,26 |
| AU4 | Brow Lowerer | 눈썹이 내려감 | 17,19,24,26 |
| AU6 | Cheek Raise | 볼(광대근)이 올라감 | 좌: 3,41,31,5<br>우: 13,46,35,11 |
| AU9 | Nose Wrinkle | 코에 주름이 생김 | 31,27,35 |
| AU12 | Lip Corner Puller | 입의 양 끝을 당김 | 좌: 3,48,6<br>우: 13,54,10 |
| AU13 | Sharp Lip Puller | 입술을 얇게 만들어 당김 | 48,54 |
| AU18 | Lip Pucker | 입술을 모아 오므림 | 48,49,50,51,52,53,54,55,56,57,58 |

### OpenCV 구현 단계

#### 1단계: 얼굴 검출
```python
import cv2
import dlib
import numpy as np

# 얼굴 검출기 초기화
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 또는 더 정확한 HOG 기반 검출기 사용
detector = dlib.get_frontal_face_detector()
```

#### 2단계: Landmark 추출
```python
# Shape predictor 모델 로드
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Landmark 좌표 추출
def get_landmarks(image, face_rect):
    landmarks = predictor(image, face_rect)
    coords = np.zeros((68, 2), dtype=int)
    
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    
    return coords
```

#### 3단계: AU 특징 계산
```python
def calculate_au_features(landmarks):
    features = {}
    
    # AU1: Inner Brow Raise
    left_inner_brow = landmarks[19:22]  # 좌측 눈썹 안쪽
    right_inner_brow = landmarks[24:26] # 우측 눈썹 안쪽
    features['AU1'] = calculate_brow_raise(left_inner_brow, right_inner_brow)
    
    # AU12: Lip Corner Puller
    left_mouth_corner = landmarks[48]
    right_mouth_corner = landmarks[54]
    features['AU12'] = calculate_lip_corner_pull(left_mouth_corner, right_mouth_corner)
    
    return features

def calculate_brow_raise(left_brow, right_brow):
    # 눈썹 높이 변화 계산 로직
    pass

def calculate_lip_corner_pull(left_corner, right_corner):
    # 입꼬리 움직임 계산 로직
    pass
```

## 3. 제스처 기반의 감성 인식 시스템

### 제스처 기반 감성 인식의 중요성

#### 비언어적 의사소통의 역할

- **표현 채널의 다양성**:
  - 표정: 얼굴 근육의 미세한 움직임
  - 제스처: 몸짓, 자세, 손동작 등

- **상호 보완적 정보**:
  - 표정만으로는 파악하기 어려운 감성의 강도
  - 제스처를 통한 감성의 맥락적 이해
  - 문화적 차이를 고려한 포괄적 인식

#### 감성별 특징적 제스처 패턴

**행복/기쁨 (Joy/Happiness)**:
- 달리기 결승선을 통과한 선수의 예시
- **표정**: 웃는 표정, 밝은 눈빛
- **제스처**: 두 팔을 벌려 승리 표현, 점프, 주먹 쥐기

**슬픔/우울 (Sadness/Depression)**:
- 슬픔에 빠진 사람의 예시
- **표정**: 내려간 눈꼬리와 입꼬리, 찡그린 이마
- **제스처**: 두 팔로 머리를 감싸는 자세, 몸을 웅크림, 고개 숙임

**분노 (Anger)**:
- **표정**: 찡그린 눈썹, 다문 입술
- **제스처**: 주먹 쥐기, 팔짱 끼기, 급격한 손동작

**놀람 (Surprise)**:
- **표정**: 둥근 눈, 벌린 입
- **제스처**: 손으로 입 가리기, 뒤로 물러서기

### 멀티모달 감성 인식의 장점

- **정확도 향상**: 단일 모달리티 대비 10-20% 성능 개선
- **강건성 증가**: 하나의 채널에 노이즈가 있어도 다른 채널로 보완
- **맥락적 이해**: 상황에 따른 감성의 미묘한 차이 파악
- **개인차 극복**: 개인별 표현 방식의 다양성 수용

## 4. OpenPose 기반 제스처 인식 구현

### OpenPose 시스템 개요

- **개발**: 카네기 멜론 대학교(CMU)
- **기능**: 실시간 다중 인물 2D 자세 추정
- **특징**: 딥러닝 기반 고정밀 관절 검출

### 관절 정보 추출

#### COCO 모델 (18개 관절점)
```
0: 코 (Nose)
1: 목 (Neck)
2: 우측 어깨 (Right Shoulder)
3: 우측 팔꿈치 (Right Elbow)
4: 우측 손목 (Right Wrist)
5: 좌측 어깨 (Left Shoulder)
6: 좌측 팔꿈치 (Left Elbow)
7: 좌측 손목 (Left Wrist)
8: 우측 엉덩이 (Right Hip)
9: 우측 무릎 (Right Knee)
10: 우측 발목 (Right Ankle)
11: 좌측 엉덩이 (Left Hip)
12: 좌측 무릎 (Left Knee)
13: 좌측 발목 (Left Ankle)
14: 우측 눈 (Right Eye)
15: 좌측 눈 (Left Eye)
16: 우측 귀 (Right Ear)
17: 좌측 귀 (Left Ear)
```

#### 구현 예시
```python
import cv2
import numpy as np

# OpenPose 모델 로드
net = cv2.dnn.readNetFromTensorflow('graph_opt.pb')

def detect_pose(image):
    # 입력 이미지 전처리
    blob = cv2.dnn.blobFromImage(image, 1.0, (368, 368), 
                                 (127.5, 127.5, 127.5), 
                                 swapRB=True, crop=False)
    
    # 네트워크에 입력
    net.setInput(blob)
    output = net.forward()
    
    # 관절점 추출
    points = extract_keypoints(output, image.shape)
    
    return points

def extract_keypoints(output, image_shape):
    H, W = image_shape[:2]
    points = []
    
    for i in range(18):  # 18개 관절점
        # 각 관절점의 확률 맵에서 최대값 위치 찾기
        probMap = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(probMap)
        
        x = (image_shape[1] * point[0]) / output.shape[3]
        y = (image_shape[0] * point[1]) / output.shape[2]
        
        if prob > 0.1:  # 임계값 이상일 때만 유효한 점으로 간주
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    return points
```

### 제스처 특징 추출

#### 각도 기반 특징
```python
def calculate_angle(p1, p2, p3):
    """세 점을 이용해 각도 계산"""
    if None in [p1, p2, p3]:
        return None
    
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def extract_gesture_features(keypoints):
    features = {}
    
    # 팔꿈치 각도
    features['left_elbow_angle'] = calculate_angle(
        keypoints[5], keypoints[6], keypoints[7])  # 좌측 어깨-팔꿈치-손목
    features['right_elbow_angle'] = calculate_angle(
        keypoints[2], keypoints[3], keypoints[4])  # 우측 어깨-팔꿈치-손목
    
    # 어깨 기울기
    if keypoints[2] and keypoints[5]:
        shoulder_slope = (keypoints[2][1] - keypoints[5][1]) / (keypoints[2][0] - keypoints[5][0])
        features['shoulder_slope'] = np.degrees(np.arctan(shoulder_slope))
    
    # 팔의 높이 (어깨 대비)
    if keypoints[4] and keypoints[2]:  # 우측 손목, 어깨
        features['right_arm_height'] = keypoints[4][1] - keypoints[2][1]
    if keypoints[7] and keypoints[5]:  # 좌측 손목, 어깨
        features['left_arm_height'] = keypoints[7][1] - keypoints[5][1]
    
    return features
```

## 5. 통합 감성 인식 시스템

### 멀티모달 특징 융합

#### 특징 수준 융합 (Feature-level Fusion)
```python
def combine_features(facial_features, gesture_features):
    combined = {}
    
    # 표정 특징
    for au, value in facial_features.items():
        combined[f'facial_{au}'] = value
    
    # 제스처 특징
    for gesture, value in gesture_features.items():
        combined[f'gesture_{gesture}'] = value
    
    return combined
```

#### 결정 수준 융합 (Decision-level Fusion)
```python
def fusion_classification(facial_prob, gesture_prob, weights=[0.6, 0.4]):
    """가중 평균을 통한 결정 융합"""
    final_prob = weights[0] * facial_prob + weights[1] * gesture_prob
    return final_prob

def adaptive_fusion(facial_confidence, gesture_confidence, facial_prob, gesture_prob):
    """신뢰도 기반 적응적 융합"""
    total_confidence = facial_confidence + gesture_confidence
    
    if total_confidence > 0:
        facial_weight = facial_confidence / total_confidence
        gesture_weight = gesture_confidence / total_confidence
        
        final_prob = facial_weight * facial_prob + gesture_weight * gesture_prob
    else:
        final_prob = 0.5 * (facial_prob + gesture_prob)
    
    return final_prob
```

### 실시간 감성 인식 파이프라인

```python
class EmotionRecognitionSystem:
    def __init__(self):
        self.face_detector = self.init_face_detector()
        self.landmark_predictor = self.init_landmark_predictor()
        self.pose_net = self.init_pose_net()
        self.emotion_classifier = self.init_emotion_classifier()
    
    def process_frame(self, frame):
        # 1. 얼굴 검출 및 표정 분석
        faces = self.face_detector(frame)
        facial_emotions = []
        
        for face in faces:
            landmarks = self.landmark_predictor(frame, face)
            au_features = self.extract_au_features(landmarks)
            facial_emotion = self.classify_facial_emotion(au_features)
            facial_emotions.append(facial_emotion)
        
        # 2. 자세 검출 및 제스처 분석
        keypoints = self.detect_pose(frame)
        gesture_features = self.extract_gesture_features(keypoints)
        gesture_emotion = self.classify_gesture_emotion(gesture_features)
        
        # 3. 멀티모달 융합
        if facial_emotions and gesture_emotion:
            final_emotion = self.fusion_classification(
                facial_emotions[0], gesture_emotion)
        elif facial_emotions:
            final_emotion = facial_emotions[0]
        elif gesture_emotion:
            final_emotion = gesture_emotion
        else:
            final_emotion = None
        
        return final_emotion
    
    def extract_au_features(self, landmarks):
        # AU 특징 추출 구현
        pass
    
    def extract_gesture_features(self, keypoints):
        # 제스처 특징 추출 구현
        pass
    
    def classify_facial_emotion(self, features):
        # 표정 기반 감성 분류
        pass
    
    def classify_gesture_emotion(self, features):
        # 제스처 기반 감성 분류
        pass
```

## 6. 성능 향상 기법

### 데이터 증강 (Data Augmentation)

- **표정 데이터**:
  - 조명 변화, 각도 변화, 노이즈 추가
  - 얼굴 크기 정규화, 히스토그램 평활화

- **제스처 데이터**:
  - 시간적 증강 (속도 변화, 구간 분할)
  - 공간적 증강 (회전, 스케일링, 이동)

### 개인화 적응

```python
class PersonalizedEmotionRecognizer:
    def __init__(self):
        self.base_model = self.load_base_model()
        self.personal_adapters = {}
    
    def adapt_to_user(self, user_id, calibration_data):
        """사용자별 개인화 적응"""
        adapter = self.train_personal_adapter(calibration_data)
        self.personal_adapters[user_id] = adapter
    
    def predict_with_adaptation(self, user_id, features):
        base_prediction = self.base_model.predict(features)
        
        if user_id in self.personal_adapters:
            adapter = self.personal_adapters[user_id]
            adapted_prediction = adapter.adapt(base_prediction, features)
            return adapted_prediction
        else:
            return base_prediction
```

## 7. 응용 분야 및 향후 발전 방향

### 주요 응용 분야

- **교육 기술**: 학습자 감정 상태 모니터링
- **헬스케어**: 환자 감정 관리 및 치료 보조
- **엔터테인먼트**: 게임, VR/AR 콘텐츠의 몰입감 향상
- **마케팅**: 소비자 반응 분석
- **보안**: 감정 기반 사용자 인증

### 기술적 도전과제

- **실시간 처리**: 저지연 감성 인식
- **프라이버시**: 개인정보 보호와 감성 인식의 균형
- **문화적 다양성**: 전 세계 다양한 표현 방식 수용
- **환경 적응성**: 다양한 조명, 각도, 거리 조건 대응

### 향후 발전 방향

- **멀티모달 확장**: 음성, 생체신호 통합
- **설명 가능한 AI**: 감성 인식 결과의 해석 가능성
- **연속적 학습**: 사용 중 지속적인 성능 개선
- **경량화**: 모바일 기기에서의 실시간 구동

## 8. 요약

- **FACS 기반 표정 분석**: 객관적이고 체계적인 얼굴 근육 움직임 분석
- **OpenCV 구현**: Landmark 기반 AU 특징 추출 및 감성 분류
- **제스처 인식**: OpenPose를 활용한 관절 정보 추출 및 자세 분석
- **멀티모달 융합**: 표정과 제스처 정보의 상호 보완적 활용
- **실용적 응용**: 다양한 HCI 분야에서의 감성 인식 시스템 구현

이러한 기술들은 인간-컴퓨터 상호작용에서 더욱 자연스럽고 직관적인 소통을 가능하게 하며, 사용자의 감성을 이해하는 지능형 시스템 구축의 핵심 기술로 활용됩니다.
