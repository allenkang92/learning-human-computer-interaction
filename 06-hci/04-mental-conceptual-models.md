# 휴먼 인터페이스: 정신 모델 & 개념 모델

## 1. 상호작용과 모델의 기본 개념

### HCI의 기초
- **정의**: 인간(지각, 인지, 기억, 주의)과 컴퓨터 시스템 간의 **인터페이스**를 통한 **상호작용**을 다루는 분야
- **궁극적 목표**: 인간의 지식과 행동을 정보로 구조화하는 것
- **핵심 요소**: 사용자, 시스템, 인터페이스, 상호작용

### 상호작용 모델의 구성 요소
- **사용자(User)**: 
  - 시스템에 대한 자신만의 **정신 모델(Mental Model)** 형성
  - 일상 경험과 지식을 바탕으로 형성됨
  
- **개발자(Developer)**: 
  - 시스템을 **개념 모형(Conceptual Model)**에 따라 설계
  - 사용자의 정신 모델을 이해하고 반영하려 노력

- **불일치 발생 시**: 
  - 사용자는 시스템 사용에 어려움을 겪음
  - 실수 발생 및 사용 효율 저하
  - 최악의 경우 시스템 사용 포기

- **이상적 상태**: 
  - 사용자의 정신 모델이 개발자의 개념 모델과 일치
  - 사용자의 행동과 언어가 개념 모델에 잘 반영되어 정보로 표현됨

### 정신 모델 vs 개념 모델

#### 정신 모델 (Mental Model, 사용자 관점)
- **정의**: 사용자가 시스템이나 대상에 대해 내적으로 가지고 있는 이해와 작동 방식에 대한 믿음
- **구성 요소**:
  - **행동(Behavior)**: 시스템 사용 방법에 대한 절차적 지식
  - **경험(Experience)**: 언어와 경험을 통해 획득한 지식
  - **습관(Habit)**: 반복적 사용으로 형성된 자동화된 행동 패턴
- **특징**:
  - 주로 **절차적 지식**으로 구성됨
  - 지각적 또는 **은유적 표상(메타포)** 통해 형성
  - 완전하거나 정확하지 않을 수 있음 (단순화된 형태로 존재)
- **노먼(Norman, 1986)의 심성 모형 세 측면**:
  - 디자인 모형: 디자이너가 의도한 시스템 모델
  - 사용자 모형: 사용자가 형성한 시스템 모델
  - 시스템 이미지: 실제 시스템 기능과 인터페이스

#### 개념 모델 (Conceptual Model, 개발자/시스템 관점)
- **정의**: 개발자가 설계한 시스템의 구조와 작동 원리, 시스템이 사용자에게 보여주는 모습
- **목적**:
  - 사용자의 정신 모델에 부합하도록 설계
  - 절차적 행동을 시각화
  - 정보를 은유적(메타포)으로 표현하여 직관적 이해 유도
- **특징**:
  - 사용자의 **행동을 끌어내고 상호작용** 유도
  - 목표(기능/서비스 이용) 달성을 위한 구조 제공
  - 시스템 인터페이스를 통해 표현됨
- **핵심 포인트**: 시스템은 사용자의 정신 모델을 반영한 개념 모델이어야 함

## 2. 정신 모델 분석 방법

### 분석 대상
- 사용자의 행동 패턴
  - 규칙 기반 행동(Rule-based)
  - 기능 기반 행동(Skill-based)
  - 지식 기반 행동(Knowledge-based)
- 습관적 행동 단계와 절차

### 정신 모델 분석 프로세스
분석은 반복적으로 수행되며 다음 단계로 구성됩니다:

1. **행동 분석**: 
   - 사용자의 논리적/습관적 행동 단계 관찰
   - 작업 수행 패턴 식별

2. **절차적 지식 파악**: 
   - 장기기억 속 행동 규칙, 조건, 습관 분석
   - 사용자의 의사결정 패턴 이해

3. **Chunking(청킹)**: 
   - 단위 행동 분류 및 그룹화
   - 관련 행동들의 묶음 파악

4. **정보 연결 (Schema)**: 
   - 행동 정보 간의 네트워크 구조화
   - 사용자 스키마 분석

5. **의미 파악 (Metaphor)**: 
   - 사용자가 사용하는 은유적 표현 분석
   - 익숙한 개념과의 연결성 파악

6. **맥락적 경험 이해**: 
   - 정보 연결의 기준 파악
   - 감성적 맥락과 사용 환경 분석

### 행동의 절차적 이해

#### 절차적 지식
- **정의**: 무의식적으로 행하는 행동에 대한 지식
- **특징**: 
  - 명확한 행동 규칙과 조건 포함
  - 반복을 통해 형성
  - '어떻게 하는지(how-to)'에 대한 지식

#### 행동 습관
- **형성 과정**: 생각과 행동의 반복으로 형성
- **특징**: 
  - 의식적 주의 없이 자동적으로 수행
  - 상황 단서에 의해 촉발됨

#### Rasmussen의 행동 모델 (1983)
1. **기능 기반 행동(Skill-based, Signal)**: 
   - 자동화된 감각운동 패턴
   - 최소한의 인지적 노력 필요
   - 예: 자전거 타기, 키보드 타이핑

2. **규칙 기반 행동(Rule-based, Sign)**: 
   - 상황 인식 후 저장된 규칙 적용
   - 논리적 판단 과정 포함
   - 예: 교통 신호에 따른 운전 행동

3. **지식 기반 행동(Knowledge-based, Symbol)**: 
   - 목표 달성을 위한 해석, 결정, 계획 필요
   - 모호하거나 새로운 상황에서 발생
   - 높은 인지적 노력 필요
   - 예: 새로운 소프트웨어 사용법 학습

**핵심 포인트**: 행동 습관과 절차적 규칙은 정신 모델의 중요 구성 요소

### 행동과 정보의 동기화
- 정보의 순차적 구조가 행동 순서를 결정
- 정보 간 연결이 행동 간 연결로 이어짐
- 행동과 정보는 상호 영향을 미치며 발전

### Chunking (청킹)
- **정의**: 관련 정보/행동을 의미 있는 단위로 묶는 과정
- **목적**: 
  - 인지적 부담 감소
  - 단기기억 용량 한계 극복
- **특징**: 정보의 계층적 구조화로 이어짐
- **예시**: 전화번호를 개별 숫자가 아닌 구역번호-국번-개별번호 묶음으로 기억

### 스키마 (Schema)
- **정의**: 정보의 연관성 네트워크, 계층적 구조
- **역할**: 
  - 지식을 구성하고 행동 유발
  - 경험을 조직화하는 인지 구조
- **특징**: 
  - 행동 스키마와 정보 스키마 간 동기화 필요
  - 사용자 작업(Task) 분석 및 체계화에 중요
- **시스템 설계 적용**: 사용자의 정신 모델 구조화에 활용

### 메타포 (Metaphor)
- **정의**: 행동의 은유적 표현
- **특징**: 
  - 언어로 표현된 심상(Image) 시각화
  - 친숙한 개념을 통해 새로운 개념 이해 지원
- **역할**: 
  - 정보를 직관적으로 이해하게 함
  - 행동을 자연스럽게 유도
- **예시**: 컴퓨터 '바탕화면', '폴더', '휴지통' 등

### 맥락적 경험
- **행동 맥락 요소**: 누가, 언제, 어디서, 무엇을, 어떻게, 왜
- **맥락의 영향**: 행동의 맥락이 정보 연결성 결정
- **행동 유도성(Affordance)**: 
  - 지각된 단서가 의식적 개입 없이 행동 유도
  - 직관적인 인터페이스 설계의 기초
- **감성적 맥락**: 
  - 감성적 정보 네트워크: 직관적 경험
  - 이성적 정보 네트워크: 논리적 경험

## 3. 개념 모델 설계 및 분석

### 개념 모델의 정의와 역할
- **정의**: 개발자를 위한 설계도이자 정신 모델을 시스템으로 구체화한 결과물
- **목적**: 
  - 실제 경험을 시스템에 옮긴 듯한 **심볼(Symbol)** 제공
  - 시스템 이해와 상호작용 촉진
- **핵심 역할**: 
  - 상호작용 도구/수단 제공
  - 메타포를 통한 직관적 이해 증진

### 개념 모델의 수준
- **추상적 수준**: 
  - 새로운 개념/혁신 시스템 이해에 중점
  - 시스템의 전반적인 구조와 철학 제시
- **구체적 수준**: 
  - 기존 시스템 향상, 행동 연결성 이해에 중점
  - 상세한 작동 방식과 인터페이스 요소 명시
- **개발 권장 방향**: 개념 위주에서 구체적 수준으로 점진적 발전

### 개념 모델의 구성 요소 (구축 관점)
1. **행동(Task)**: 
   - 목적, 요구사항, 기능, 환경을 포함
   - 사용자의 절차적 지식과 일치하는 행위 순서
   - 사용자 관찰을 통한 행동 패턴 이해가 중요

2. **정보(Information)**: 
   - 범주화/계층화로 인지적 부담 감소
   - 목적/요구사항을 기능/서비스로 시각화
   - 사용자가 이해하기 쉬운 형태로 구조화

3. **상호작용(Interaction)**: 
   - 정보 인지에서 행동 실행까지의 과정
   - 사용자 행동 유도 → 결과 제시 → 다음 행동 정보 제시
   - 직관적이고 자연스러운 흐름 설계

### 개념 모델 시각화 방법
- **목록과 표(List and Tables)**: 
  - 정보의 체계적 구조화
  - 항목 간 관계 명시

- **도표(Diagrams)**: 
  - 정보 흐름과 관계 시각화
  - 시스템 구조 명시

- **스토리보드/스케치(Storyboard and Sketches)**: 
  - 시나리오 기반 정보 흐름 표현
  - 사용자 여정과 상호작용 시각화

- **설명서(Written descriptions)**: 
  - 단계별 상세 설명
  - 개념과 프로세스 문서화

- **무드보드(Mood Boards)**: 
  - 이미지/스타일로 디자인 컨셉 제시
  - 감성적 요소 표현

### 개념 모델 개발을 위한 조사 및 분석

#### 사용자 정의 및 조사
- **페르소나(Persona)**: 
  - 가상의 대표 사용자 프로필
  - 사용자 그룹 특성(환경, 사회, 문화, 신체, 인지 요인) 분석 기반
  - 보통 3-5명의 페르소나 설정
  - 프로필, 태도, 니즈, 목표, 시나리오 등 포함

#### 행동 분석 방법론
- **시나리오 분석**: 
  - 사용자 경험 기반 시나리오 구성
  - 시나리오를 통한 사용자 행동 분석
  - 개발 초기 단계에 특히 유용

- **Norman의 행동 모델(7단계)**: 
  1. 목표 설정
  2. 의도 형성
  3. 행위 계획
  4. 행위 실행
  5. 시스템 반응 지각
  6. 반응 확인(해석)
  7. 반응 평가(목표/의도와 비교)
  - 목표: 실행의 간극 / 평가의 간극 최소화

- **작업 분석(Task Analysis)**: 
  - 목표 달성을 위한 행동 과정 분석
  - 외현적 행동 + 내적 의도 파악 필요
  - 주요 방법:
    - **계층적 작업 분석(HTA)**: 작업을 세부 작업/단위 동작으로 분해
    - **지식 기반 분석**: 도구-행위 매핑 및 분류체계 구축
    - **시퀀스 모델**: 세부 작업의 순차적 기술(도형 활용), 단계별 의도/이유/충돌 명시

#### 정보 디자인
- **친화 도법(Affinity Diagram)**: 
  - 정보 수집 → 대표 표제화 → 범주화 → 체계화(계층 구조)
  - 청킹(Chunking) 원칙 적용
  - 추상적 현상을 시스템 작동 가능한 정보 구조로 변환

- **상호작용을 위한 정보 디자인 원칙**:
  1. **행동 유도성(Affordance)**: 
     - 사물의 지각적/고유 특성이 특정 행동을 유도
     - 정보가 직관적 행동을 이끌어내는 속성

  2. **매핑(Mapping)**: 
     - 조작(정보)과 결과(사물) 간 관계의 명확성/일치성
     - 예: 스토브 버튼과 화구 간의 공간적 대응 관계

  3. **제약(Constraint)**: 
     - 실수 방지를 위한 물리적/논리적 제한
     - 사용자 행동의 가능성 범위 설정

  4. **피드백(Feedback)**: 
     - 사용자 요청에 대한 시스템 반응 정보
     - 행동의 결과를 즉각적으로 알려주는 기능

- **메타포 활용**: 
  - 사용자 직관/경험(습관, 심상)에 호소
  - 형태(표면), 기능/구성요소, 목적이 사용자 경험과 일치해야 효과적
  - 개발 과정: 사용자 목적/행동/맥락 분석 → 기존 메타포 참조 → 디자인 → 테스트

- **메타포와 모델 관계**: 
  - 메타포는 사용자의 정신 모델과 개발자의 개념 모델을 연결하는 다리 역할
  - 인터페이스를 통해 표현 언어/행위 언어로 구체화

#### 프로토타이핑(Prototyping)
- **정의**: 개념 모델을 구체적으로 작동 가능하게 구현하는 과정
- **종류**: 
  - 시각적 제시형(애니메이션/동영상)
  - 직접 조작 가능형(기능 모델)
- **개발 기반**: 화면 스케치/스토리보드 등으로 발전
- **목적**: 사용자 평가를 통한 시스템 개선 및 만족도 향상

## 4. 핵심 요약

- **성공적 HCI의 조건**: 사용자의 **정신 모델**과 시스템의 **개념 모델** 간 **일치**
- **개발자의 역할**: 사용자 정신 모델을 이해하고 이를 직관적인 개념 모델로 구현
- **중요 분석/설계 도구**:
  - 페르소나
  - 작업 분석
  - 정보 디자인 원칙(행동 유도성, 매핑, 제약, 피드백)
  - 메타포
  - 프로토타이핑
- **궁극적 목표**: 사용자가 시스템을 통해 자신의 목표를 쉽고 효과적으로 달성하도록 지원
