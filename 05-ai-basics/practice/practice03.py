# 데이터셋
# 총 12개 열로 구성된 데이터로서 첫 번째 열은 피험자 ID, 두 번째 열은 성별,
# 세 번째 열은 해당 감성, 네 번째 열부터 열두 번째 열까지 심전도 신호로부터 추출된 변수들에 해당
# 총 120명의 데이터, 남성 60명, 여성 60명으로서 감성은 총 세 가지로 분류된 감성으로
# 이완, 중립, 각성으로 각 40명씩 수집된 데이터로 구성되어 있다.
# 이완일 때는 약 65bpm 미만, 중립일 때는 70대 bpm 심박수를, 각성일 때는 90 이상의 심박 수를 보이고 있다.


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
            # 주의: 아래 코드에 오류가 있음 - 항상 cluster_info[x] == 1 조건만 검사함
            # 올바른 코드는 cluster_info[x] == i 여야 함
            index = list(filter(lambda x: cluster_info[x] == i, range(len(cluster_info))))
            
            # 필터링된 인덱스 중에서 random_num개만큼 무작위 선택
            # 그 인덱스들에 해당하는 데이터값을 Samples에 추가
            for index in random.sample(index, random_num):
                Samples.append(data[index])
    
    # 추출할 표본 수가 선택된 군집 수로 나누어 떨어지지 않는 경우
    # 이 부분은 구현이 완성되지 않았음 (주석 처리된 부분)
    # else:
    #     # 각 군집별 샘플 수 계산 로직
    #     # 각 군집에서 표본 추출
    #     # 남은 표본 수 처리
    
    # 결과 출력: 표본 크기와 추출된 표본 값들
    print("Cluster Samples, N = {} : {}".format(n, Samples))


if __name__ == "__main__":
    main()