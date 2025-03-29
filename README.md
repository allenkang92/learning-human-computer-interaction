# Human-Computer Interaction Learning Repository

인간-컴퓨터 상호작용(HCI)에 대해서 학습한 내용을 담으려 노력하려 합니다. 개인 아카이빙이라고 말씀드릴 수 있을 것 같아요.
인지심리학, 인공지능, 사용자 경험(UX) 등 HCI와 관련된 다양한 주제를 다뤄보려고 합니다.

## 목차

### 01. HCI 기본 개념
- [1.1 HCI의 기초](01-foundations/01-hci-fundamentals.md): HCI 개요, 중요성, 목표, 다양한 관점
- [1.2 HCI의 발전 및 적용](01-foundations/02-hci-evolution-application.md): 개념적 변화, 광범위한 적용 범위
- [1.3 HCI의 학문적 측면](01-foundations/03-hci-academic-aspects.md): 관련 학문 분야, 미래 전망

### 02. 인지와 감성
- [2.1 인지심리학과 인공지능](02-cognition-emotion/01-cognitive-ai.md): 심리학과 인공지능의 관계, 계산 모델
- [2.2 인지 과정과 뇌 연구](02-cognition-emotion/02-cognitive-brain-research.md): 인지심리학과 뇌 연구 개요
- [2.3 얼굴 표정과 감성 인식](02-cognition-emotion/03-facial-expressions-emotions.md): FACS, 감성 표현, 표정 인식 기술
- [2.4 생체 신호 기반 감성 인식](02-cognition-emotion/04-physiological-emotion-recognition.md): 심장 신호, HRV 분석, 생체 신호 기반 감성 인식
- [2.5 감성 컴퓨팅](02-cognition-emotion/05-emotional-computing.md): 감성의 필요성, 감성의 정의와 분류, 감성 반응 이론
- [2.6 감성 반응의 생체공학](02-cognition-emotion/06-emotional-bioengineering.md): 인간 정보 처리, 중추 신경계, 자율 신경계 반응

### 03. 인간 지각과 인지
- [3.1 인간 정보 처리 모형](03-human-perception-cognition/01-information-processing-model.md): 인간의 정보 처리 과정, 구성 요소
- [3.2 시각적 지각](03-human-perception-cognition/02-visual-perception.md): 눈의 구조와 기능, 시각 능력, 시지각 체제화 원리
- [3.3 청각적 지각](03-human-perception-cognition/03-auditory-perception.md): 귀의 구조와 기능, 음의 성질과 측정, 청각 정보 처리
- [3.5 지각 측정 및 정신물리학](03-human-perception-cognition/05-perception-measurement.md): 물리량과 지각량의 관계, 측정법, 신호 탐지 이론
- [3.6 신경 시스템](03-human-perception-cognition/06-nervous-system.md): 뉴런의 구조와 기능, 신경계의 구성, 뇌의 주요 구조와 기능

### 04. 사용자 경험

### 05. AI와 머신러닝
- [5.1 비지도학습과 K-평균 군집화](05-ai-basics/01-unsupervised-learning-kmeans.md): 비지도학습 개념, K-평균 알고리즘 원리, 응용 사례
- 실습:
  - [지도학습 - KNN을 활용한 감성 분류](05-ai-basics/practice/practice01.py): K-최근접 이웃 알고리즘을 사용한 감성 데이터 분류
  - [비지도학습 - K-평균 군집화](05-ai-basics/practice/practice02.py): K-평균 군집화를 통한 감성 데이터 군집 분석

### 06. 인간 인지 과정과 HCI
- [6.1 단기 기억](06-hci/01-short-term-memory.md): 인간 정보 처리 모형, 단기 기억 특성, 용량 한계, 부호화
- [6.2 주의](06-hci/02-attention.md): 주의의 특성, 선택적 주의, 상향식/하향식 처리, 주의 집중과 분산
- [6.3 장기 기억](06-hci/03-long-term-memory.md): 장기 기억 특성, 정교화, 인출, 스키마, 기억 종류

## 디렉토리 구조

```
learning-human-computer-interaction/
├── 01-foundations/               # HCI 기본 개념 관련 자료
│   ├── 01-hci-fundamentals.md    # HCI 기초 개념
│   ├── 02-hci-evolution-application.md  # HCI 발전 및 적용
│   └── 03-hci-academic-aspects.md       # HCI 학문적 측면
│
├── 02-cognition-emotion/         # 인지와 감성 관련 자료
│   ├── 01-cognitive-ai.md        # 인지심리학과 인공지능
│   ├── 02-cognitive-brain-research.md  # 인지 과정과 뇌 연구
│   ├── 03-facial-expressions-emotions.md  # 얼굴 표정과 감성 인식
│   ├── 04-physiological-emotion-recognition.md  # 생체 신호 기반 감성 인식
│   ├── 05-emotional-computing.md  # 감성 컴퓨팅
│   └── 06-emotional-bioengineering.md  # 감성 반응의 생체공학
│
├── 03-human-perception-cognition/  # 인간 지각과 인지 관련 자료
│   ├── 01-information-processing-model.md  # 인간 정보 처리 모형
│   ├── 02-visual-perception.md    # 시각적 지각
│   ├── 03-auditory-perception.md  # 청각적 지각
│   ├── 05-perception-measurement.md  # 지각 측정 및 정신물리학
│   └── 06-nervous-system.md       # 신경 시스템
│
├── 04-user-experience/           # 사용자 경험 관련 자료
│
├── 05-ai-basics/                 # AI 및 머신러닝 관련 자료
│   ├── 01-unsupervised-learning-kmeans.md  # 비지도학습과 K-평균 군집화
│   └── practice/                 # 실습 자료
│       ├── practice01.py         # 지도학습 - KNN을 활용한 감성 분류
│       └── practice02.py         # 비지도학습 - K-평균 군집화
│
└── 06-hci/                       # 인간 인지 과정과 HCI 관련 자료
    ├── 01-short-term-memory.md   # 단기 기억
    ├── 02-attention.md           # 주의
    └── 03-long-term-memory.md    # 장기 기억
```
