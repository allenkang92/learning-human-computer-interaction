# 데이터셋
# 총 12개 열로 구성된 데이터로서 첫 번째 열은 피험자 ID, 두 번째 열은 성별,
# 세 번째 열은 해당 감성, 네 번째 열부터 열두 번째 열까지 심전도 신호로부터 추출된 변수들에 해당
# 총 120명의 데이터, 남성 60명, 여성 60명으로서 감성은 총 세 가지로 분류된 감성으로
# 이완, 중립, 각성으로 각 40명씩 수집된 데이터로 구성되어 있다.
# 이완일 때는 약 65bpm 미만, 중립일 때는 70대 bpm 심박수를, 각성일 때는 90 이상의 심박 수를 보이고 있다.

# ============ 감성인식 알고리즘을 위한 실험 설계 ============
# 1. 실험 설계 개요: 감성 유발 실험
#    - 정의: 피실험자(Participant/Subject)에게 목표 감성을 유발하기 위해 시각, 청각, 촉각 등 
#           단일 또는 복합 자극을 사용하는 실험
#    - 목표: 특정 감성을 안정적으로 유발하는 자극을 찾아내고, 그 과정에서 나타나는 
#           생리적/행동적 반응을 측정하여 감성 인식 알고리즘 개발에 활용
#
# 2. 감성 유발 자극 결정 과정 (5단계)
#    1) 감성 유발 자극 수집 (Stimulus Collection):
#       - 목표 감성을 유발할 가능성이 있는 자극 후보군을 최대한 많이 수집
#       - 종류: 시각 자극(사진, 비디오), 청각 자극(음악, 화이트 노이즈), 복합 자극(시각+청각)
#       - 주의: 초기에는 어떤 자극이 효과적일지 모르므로 다양한 후보군 확보가 중요
#    2) 감성 유발 실험 (Emotion Induction Experiment):
#       - 수집된 후보 자극을 피실험자에게 노출시키고 반응을 측정
#       - 연구 설계에 따라 자극 제시 순서, 횟수 등을 조절하며 반복 수행
#       - 휴식 시간 부여: 이전 자극의 영향에서 피실험자가 회복하여 다음 자극에 대한 
#         순수한 반응을 얻기 위함 (너무 길면 집중력 저하, 너무 짧으면 이전 감성 영향 잔존)
#    3) 주관 평가 (Subjective Evaluation):
#       - 피실험자가 실제로 목표 감성을 느꼈는지 설문 등을 통해 검증
#       - 예시: 감성 평가 척도(SAM: Self-Assessment Manikin 등) 사용
#    4) 통계 분석 (Statistical Analysis):
#       - 수집된 주관 평가 데이터와 생체 신호 데이터 등을 통계적으로 분석
#       - 어떤 자극이 목표 감성을 유의미하게 유발했는지 확인
#    5) 감성 유발 자극 결정 (Stimulus Decision):
#       - 통계 분석 결과, 유의미하게 목표 감성을 유발하는 것으로 나타난 자극들을 최종적으로 선정
#
# 3. 피실험자(Participant/Subject) 선정
#    - 중요성: 데이터 수집의 핵심 역할. 연구 결과의 질을 좌우
#    - 고려사항:
#      - 연구 분야와 가설에 적합한 모집단 선정(특정 연령대, 성별 등)
#      - 통계적 유의성을 확보할 수 있는 충분한 인원수 확보
#    - 표본 집단 사용 이유:
#      - 현실적 제약: 전체 모집단을 대상으로 실험하는 것은 시간과 비용 측면에서 거의 불가능
#      - 해결책: 모집단을 대표할 수 있는 표본 집단을 추출하여 실험 진행
#      - 필수 조건: 표본 집단은 편향되지 않아야 모집단의 특성을 잘 반영할 수 있음
#
# 4. 표본 추출 방법 (Sampling Methods)
#    본 코드는 아래 표본 추출 방법들을 구현하고 있음
#    - 확률적 표본 추출 (Probabilistic Sampling):
#      1) 단순 무작위 표본 (Simple Random Sampling)
#      2) 층화 표본 (Stratified Random Sampling)
#      3) 체계적 표본 (Systematic Sampling)
#      4) 군집 표본 (Cluster Sampling)
# =========================================================


# 판다스를 통해서 데이터를 로드
import pandas as pd

def main():
    # CSV 파일 로드 - 인코딩은 UTF-8 사용
    df = pd.read_csv("/Users/ddang/learning-human-computer-interaction/learning-human-computer-interaction/data/Chapter3_data.CSV", encoding = 'utf8')

    # 로드되는 데이터 중에서 네 번째 열인 heart rate(심박수) 데이터만 추출
    # iloc[:, [3][0]]는 모든 행(:)에서 네 번째 열([3][0])의 값을 선택함
    # values로 numpy 배열로 변환 후 tolist()로 파이썬 리스트로 변환
    X = df.iloc[:, [3][0]].values
    X = X.tolist()

    # 감성 카테고리 정보(0: 이완, 1: 중립, 2: 각성) 추출
    # 층화 표본 추출과 군집 표본 추출에서 층화/군집 정보로 사용
    Emotions = df.iloc[:, [2][0]].values
    Emotions = Emotions.tolist()

    # 각 표본 추출 방법 호출 및 테스트
    Simple_Random_Samples(X, 10)        # 심박수 데이터에서 10개 단순 무작위 추출
    Stratified_Random_Samples(X, Emotions, 10)  # 감성별로 균등하게 10개 층화 추출
    Systematic_Samples(X, 30, 10)       # 30 간격으로 10개 체계적 추출
    Cluster_Samples(X, Emotions, [1, 2], 10)    # 감성 1, 2 군집에서 10개 추출

import random

# 단순 무작위 표본 추출 (Simple Random Sampling)
# 파라미터:
#   - data: 표본을 추출할 원본 데이터 리스트
#   - n: 추출할 표본의 개수
# 반환값:
#   - 없음 (추출된 표본을 출력만 함)
# 기능:
#   - 전체 모집단에서 무작위로 n개의 표본을 선택하는 가장 기본적인 표본 추출 방법
#   - 모집단의 모든 개체가 선택될 확률이 동일함
#   - random.sample()을 사용하여 중복 없이 n개의 무작위 표본 추출
# 감성인식 실험에서의 활용:
#   - 특정 감성별 분류 없이 전체 피실험자 모집단에서 무작위로 샘플링하여 다양한 감성 반응 수집 시 사용
#   - 연구 초기 탐색적 분석 단계에서 기준 데이터(baseline) 수집에 적합
def Simple_Random_Samples(data, n):
    # random.sample() 함수를 사용하여 data에서 중복 없이 n개 항목 무작위 선택
    result = random.sample(data, n)
    # 결과 출력: 표본 크기와 추출된 표본 값들
    print("Simple random samples, N = {} : {}".format(n, result))

# 층화 표본 추출 (Stratified Random Sampling)
# 파라미터:
#   - data: 표본을 추출할 원본 데이터 리스트
#   - Stratified_info: 각 데이터 항목의 계층(층) 정보를 담은 리스트
#   - n: 추출할 총 표본의 개수
# 반환값:
#   - 없음 (추출된 표본을 출력만 함)
# 기능:
#   - 모집단을 특성(이 경우 감성 상태)에 따라 여러 층(strata)으로 나눈 후, 
#   - 각 층에서 비례적으로 표본을 추출하는 방법
#   - 모집단의 특성별 분포를 표본에 반영할 수 있음
# 감성인식 실험에서의 활용:
#   - 이완(0), 중립(1), 각성(2) 등 감성 상태별로 균등하게 피실험자를 선별해야 할 때 유용
#   - 성별, 연령대 등 인구통계학적 특성을 균형 있게 표본에 포함시킬 때 사용
#   - 감성 알고리즘의 다양한 감성 상태에 대한 일반화 성능 확보에 중요
def Stratified_Random_Samples(data, Stratified_info, n):
    # 추출된 표본을 담을 빈 리스트 초기화
    Samples = []
    # 고유한 층(strata) 식별자들의 집합 생성 (중복 제거)
    # Emotions의 경우 {0, 1, 2} (이완, 중립, 각성)이 될 것임
    stratified_types = set(Stratified_info)

    # 추출할 표본 수가 층의 개수로 나누어 떨어지는 경우
    # 예: 총 10개 표본, 3개 층 -> 각 층에서 3개씩 추출하고 남은 1개는 별도 처리
    if n % len(stratified_types) == 0:
        # 각 층에서 추출할 표본 수 계산
        random_num = int(n / len(stratified_types))
        
        # 각 층(strata)마다 반복
        for i in stratified_types:
            # 현재 층에 해당하는 데이터의 인덱스만 필터링하여 리스트 생성
            # 예: 감성이 1(중립)인 데이터들의 인덱스만 추출
            index = list(filter(lambda x : Stratified_info[x] == i, range(len(Stratified_info))))
            
            # 필터링된 인덱스 중에서 random_num개만큼 무작위 선택
            # 그 인덱스들에 해당하는 데이터값을 Samples에 추가
            for index in random.sample(index, random_num):
                Samples.append(data[index])
    
    # 추출할 표본 수가 층의 개수로 나누어 떨어지지 않는 경우
    else:
        # 기본적으로 각 층에서 균등하게 추출할 개수 계산
        random_num = int(n / len(stratified_types))
        
        # 각 층에서 random_num개씩 추출
        for i in stratified_types:
            # 현재 층에 해당하는 데이터의 인덱스 필터링
            index = list(filter(lambda x : Stratified_info[x] == i, range(len(Stratified_info))))
            # 필터링된 인덱스에서 random_num개 무작위 선택 후 표본에 추가
            for index in random.sample(index, random_num):
                Samples.append(data[index])
        
        # 남은 표본 수를 채우기 위해 전체 데이터에서 무작위로 1개 추가 추출
        Samples.append(data[random.randint(0, len(data) - 1)])  # -1 추가하여 인덱스 범위 오류 방지

    # 결과 출력: 표본 크기와 추출된 표본 값들
    print("Straitified Random Samples, N = {} : {}".format(n, Samples))

# 체계적 표본 추출 (Systematic Sampling)
# 파라미터:
#   - data: 표본을 추출할 원본 데이터 리스트
#   - interval: 표본 추출 간격
#   - n: 추출할 표본의 개수
# 반환값:
#   - 없음 (추출된 표본을 출력만 함)
# 기능:
#   - 첫 번째 항목을 무작위로 선택한 후, 일정한 간격으로 표본을 추출하는 방법
#   - 표본 추출의 편향을 줄이고 분포를 고르게 유지할 수 있음
#   - 특히 시간/공간적으로 정렬된 데이터에서 유용함
# 감성인식 실험에서의 활용:
#   - 시간에 따른 연속적인 생체 신호 데이터(ECG, PPG 등)에서 일정 간격으로 데이터 추출 시 적합
#   - 장시간 실험에서 일정 간격으로 피실험자의 감성 상태를 표집할 때 유용
#   - 자극 제시 후 특정 시간 간격으로 반응을 측정하는 실험 설계에 활용
def Systematic_Samples(data, interval, n):
    # 추출된 표본을 담을 빈 리스트 초기화
    Samples = []
    # 시작 인덱스를 0부터 데이터 길이 사이에서 무작위로 선택
    index = random.randint(0, len(data) - 1)  # -1 추가하여 인덱스 범위 오류 방지

    # n개의 표본을 추출할 때까지 반복
    for i in range(0, n):
        # 현재 인덱스가 데이터 범위 내에 있는 경우
        if len(data) > index:
            # 현재 인덱스의 데이터 값을 표본에 추가
            Samples.append(data[index])
            # 다음 표본을 위해 인덱스에 간격을 더함
            index += interval
        # 인덱스가 범위를 벗어난 경우 (배열 끝에 도달)
        else:
            # 새로운 시작점을 무작위로 다시 선택
            index = random.randint(0, len(data) - 1)
            # 새 위치의 데이터 값을 표본에 추가
            Samples.append(data[index])
            # 다음 표본을 위해 인덱스에 간격을 더함
            index += interval
    
    # 결과 출력: 표본 크기와 추출된 표본 값들
    print("Systematic Samples, N = {} : {}".format(n, Samples))


# 군집 표본 추출 (Cluster Sampling)
# 파라미터:
#   - data: 표본을 추출할 원본 데이터 리스트
#   - cluster_info: 각 데이터 항목의 군집 정보를 담은 리스트
#   - option: 군집 선택 옵션. -1이면 모든 군집 사용, 리스트면 해당 군집만 사용
#   - n: 추출할 표본의 개수
# 반환값:
#   - 없음 (추출된 표본을 출력만 함)
# 기능:
#   - 모집단을 여러 군집으로 나누고, 일부 군집만 선택하여 그 안에서 표본 추출
#   - 각 군집은 모집단의 특성을 대표해야 함
#   - 지리적으로 분산된 모집단 조사 시 효율적
# 감성인식 실험에서의 활용:
#   - 특정 감성 군집(예: 중립과 각성 상태)에 초점을 맞춘 연구에 적합
#   - 특정 실험 조건이나 피실험자 그룹에 대한 집중 분석 시 유용
#   - 감성에 따른 생리 신호 패턴 비교 연구에 활용
def Cluster_Samples(data, cluster_info, option, n):
    # 추출된 표본을 담을 빈 리스트 초기화
    Samples = []
    
    # 군집 선택 옵션에 따라 사용할 군집 결정
    # option이 -1이면 모든 군집 유형 사용
    if option == -1:
        cluster_types = set(cluster_info)  # 모든 고유 군집 ID
    # 그 외의 경우, 주어진 option 리스트에 있는 군집만 사용
    else:
        cluster_types = option  # 예: [1, 2]는 감성이 중립과 각성인 군집만 사용
    
    # 추출할 표본 수가 선택된 군집 수로 나누어 떨어지는 경우
    # 각 군집에서 동일한 수의 표본 추출
    if n % len(cluster_types) == 0:
        # 각 군집에서 추출할 표본 수 계산
        random_num = int(n / len(cluster_types))
        
        # 선택된 각 군집에 대해 반복
        for i in cluster_types:
            # 현재 군집(i)에 해당하는 데이터의 인덱스만 필터링
            index = list(filter(lambda x: cluster_info[x] == i, range(len(cluster_info))))
            
            # 필터링된 인덱스 중에서 random_num개만큼 무작위 선택
            # 그 인덱스들에 해당하는 데이터값을 Samples에 추가
            for index in random.sample(index, random_num):
                Samples.append(data[index])
    
    # 추출할 표본 수가 선택된 군집 수로 나누어 떨어지지 않는 경우
    else:
        # 나누어 떨어지지 않는 경우의 처리 로직 추가
        # 각 군집에서 균등하게 추출할 기본 개수 계산
        random_num = n // len(cluster_types)
        # 남은 표본 수 계산
        remainder = n % len(cluster_types)
        
        # 각 군집에서 기본 개수만큼 추출
        for i in cluster_types:
            # 현재 군집에 해당하는 데이터의 인덱스 필터링
            index_list = list(filter(lambda x: cluster_info[x] == i, range(len(cluster_info))))
            
            # 이 군집에서 추출할 실제 개수 결정 (남은 표본 수 고려)
            actual_num = random_num + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1
                
            # 필터링된 인덱스 중에서 actual_num개 무작위 선택
            for index in random.sample(index_list, min(actual_num, len(index_list))):
                Samples.append(data[index])
    
    # 결과 출력: 표본 크기와 추출된 표본 값들
    print("Cluster Samples, N = {} : {}".format(n, Samples))


# 감성 유발 자극 평가를 위한 실험 설계 함수
def design_emotion_induction_experiment(stimuli_list, participant_count, emotion_types):
    """
    감성 유발 자극 평가를 위한 실험 설계를 수행합니다.
    
    파라미터:
        - stimuli_list: 평가할 자극 목록 (예: 사진, 소리, 동영상 등의 파일명 리스트)
        - participant_count: 필요한 피실험자 수
        - emotion_types: 유발하려는 감성 유형 목록 (예: ['이완', '중립', '각성'])
        
    반환값:
        - 실험 설계 정보를 포함한 딕셔너리
    """
    # 실험 설계 정보를 담을 딕셔너리
    experiment_design = {
        'stimuli': stimuli_list,
        'target_emotions': emotion_types,
        'required_participants': participant_count,
        'experiment_phases': [
            '1. 감성 유발 자극 수집',
            '2. 감성 유발 실험 수행',
            '3. 주관 평가 수집',
            '4. 통계 분석',
            '5. 최종 감성 유발 자극 선정'
        ],
        'sampling_methods': [
            'Simple Random Sampling',
            'Stratified Random Sampling',
            'Systematic Sampling',
            'Cluster Sampling'
        ]
    }
    
    print("\n===== 감성 유발 실험 설계 정보 =====")
    print(f"대상 감성: {emotion_types}")
    print(f"필요 피실험자 수: {participant_count}명")
    print(f"평가할 자극 수: {len(stimuli_list)}개")
    print("실험 단계:")
    for phase in experiment_design['experiment_phases']:
        print(f"  - {phase}")
    print("활용 가능한 샘플링 방법:")
    for method in experiment_design['sampling_methods']:
        print(f"  - {method}")
    
    return experiment_design


if __name__ == "__main__":
    main()
    
    # 감성 유발 실험 설계 예시 실행
    sample_stimuli = ['relaxing_music.mp3', 'neutral_image.jpg', 'exciting_video.mp4']
    sample_emotions = ['이완', '중립', '각성']
    design_emotion_induction_experiment(sample_stimuli, 30, sample_emotions)