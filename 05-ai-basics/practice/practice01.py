# 머신 러닝 기법 중 지도학습인 KNN을 통해 감성 분류.

# 데이터의 첫 번째 열은 피험자 ID(A~Z)
# 두 번째 열은 각성도, 각성도는 1일수록 각성, 0에 가까워질수록 이완.
# 세 번째 열은 긍정도, 긍정도는 1일수록 긍정, 0에 가까워질수록 부정.
# 마지막 열은 각성도와 긍정도에 의한 네 가지로 나뉜 그룹 정보.
# 그룹 1은 각성도와 긍정도가 모두 0.5 이상인 그룹이고,
# 그룹 2는 각성도는 0.5보다 작고, 긍정도는 0.5보다 크다.
# 그룹 3은 각성도와 긍정도 모두 0.5보다 작다.
# 그룹 4는 각성도는 0.5보다 크며, 긍정도는 0.5보다 작다.

# 필요한 라이브러리 임포트하기.
import pandas as pd  # 데이터 처리를 위한 판다스 라이브러리.
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 맷플롯립 라이브러리.
from sklearn.neighbors import KNeighborsClassifier  # KNN 분류 알고리즘 사용을 위한 사이킷런 라이브러리.

def main():
    print("main")  # 메인 함수 실행 시작을 알리는 출력 역할.

    #  데이터 로드 _ CSV 파일
    df = pd.read_csv('data/Chapter1_data.CSV', encoding = 'utf8')

    # 특성(feature) 데이터: 각성도와 긍정도.
    X = df.iloc[:, [1, 2]].values
    # 레이블(label) 데이터: 그룹 정보.
    group = df.iloc[:, [3][0]].values

    # KNN 분류기 초기화 (k=5, 즉 5개의 가장 가까운 이웃을 고려)
    classifier = KNeighborsClassifier(n_neighbors = 5)

    # 훈련 데이터 설정하기.
    training_data = X
    training_label = group

    # 모델 훈련: 각성도, 긍정도 데이터로 감성 그룹 분류 학습.
    classifier.fit(training_data, training_label)
    
    # 테스트 데이터 설정: 새로운 각성도, 긍정도 값에 대한 그룹 예측
    test_data = [
        [0.2, 0.1],  # 낮은 각성도, 낮은 긍정도 (그룹 3 예상)
        [0.4, 0.7],  # 낮은 각성도, 높은 긍정도 (그룹 2 예상)
        [0.5, 0.8],  # 중간 각성도, 높은 긍정도 (그룹 1 또는 2 예상)
    ]

    # 테스트 데이터에 대한 그룹 예측.
    test_result = classifier.predict(test_data)

    # 예측 결과 출력.
    print("test_result:", test_result)

    # 모든 학습 데이터 포인트를 검은색으로 산점도에 표시.
    plt.scatter(X[:,0], X[:,1], c = 'black', label = 'person')
    # x축 라벨 설정: 각성도
    plt.xlabel('Arousal')
    # y축 라벨 설정: 긍정도
    plt.ylabel('Valence')
    # 범례 표시
    plt.legend()

    # 훈련 데이터를 그룹별로 다른 색상으로 표시
    for number, i in enumerate(X):
        if (training_label[number] == 1):
            plt.scatter(i[0], i[1], c = 'blue')  # 그룹 1: 파란색 (높은 각성도, 높은 긍정도)
        if (training_label[number] == 2):
            plt.scatter(i[0], i[1], c = 'yellow')  # 그룹 2: 노란색 (낮은 각성도, 높은 긍정도)
        if (training_label[number] == 3):
            plt.scatter(i[0], i[1], c = 'red')  # 그룹 3: 빨간색 (낮은 각성도, 낮은 긍정도)
        if (training_label[number] == 4):
            plt.scatter(i[0], i[1], c = 'brown')  # 그룹 4: 갈색 (높은 각성도, 낮은 긍정도)

    # 테스트 데이터 포인트를 예측된 그룹에 따라 색상 지정하여 표시
    for i, t in enumerate(test_data):
        if (test_result[i] == 1):
            plt.scatter(t[0], t[1], c = 'blue')  # 그룹 1 예측: 파란색
        if (test_result[i] == 2):
            plt.scatter(t[0], t[1], c = 'yellow')  # 그룹 2 예측: 노란색
        if (test_result[i] == 3):
            plt.scatter(t[0], t[1], c = 'red')  # 그룹 3 예측: 빨간색
        if (test_result[i] == 4):
            plt.scatter(t[0], t[1], c = 'brown')  # 그룹 4 예측: 갈색

    # 그래프 화면에 표시
    plt.show()
    # 그래프 창 닫기 (메모리 정리를 위함.)
    plt.close()

# 스크립트가 직접 실행될 때 main() 함수 호출
if __name__ == "__main__":
    main()