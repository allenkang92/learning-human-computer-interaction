# 라이브러리 호출
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 메인을 호출하기 위한 함수
def main():
    print('main')

    df = pd.read_csv('data/Chapter2_data.CSV', encoding = 'utf8')
    # X라는 식별자에 각성도와 긍정도 데이터만을 저장
    X = df.iloc[:, [1, 2]].values

    test_data = [
        [0.2, 0.1],  # 낮은 각성도, 낮은 긍정도 
        [0.4, 0.7],  # 낮은 각성도, 높은 긍정도 
        [0.5, 0.8],  # 중간 각성도, 높은 긍정도 
    ]
    print(x) # 각성도와 긍정도가 제대로 저장되었는지 확인

    # 클러스터 모델을 위해서 5개 정보 설정
    # 클러스터 셋팅
    # 1. 클러스터 K를 설정
    _k = 4

    # 2. 중심값 설정 방법 중, K-means+사용
    _cluster_init = 'k-means++'
    
    # 3. 중심값을 찾기 위한 반복 횟수와 
    # 최대 반복 횟수 설정 및 반복을 종료하기 위한 기준값 설정
    _iter_count  = 100
    _max_iter = 300
    _tol = 0.0001

    # 트레이닝 데이터를 사용하여 K-means model fit
    kmeans_model = KMeans(n_clusters = _k, init = _cluster_init, n_init = _iter_count, tol = _tol)

    kmeans_model.fit(X)
    # 트레이닝 데이터를 확인(어떻게 군집이 되었는지 kmeans model 라벨)
    print(kmeans_model.labels_)

    # 클러스터 모델을 사용하여 현재 클러스터의 중심값을 final centroid 값에 할당
    final_centroid = kmeans_model.cluster_centers_
    print(final_centroid)

    cluster_index = TestResult(final_centroid, test_data)

    # 각 군집된 데이터들에 대한 시각화 구현
    for number, i in enumerate(X):
        if (kmeans_model.labels_[number] == 0):
            plt.scatter(i[0], i[1], c = 'red')
        elif (kmeans_model.labels_[number] == 1):
            plt.scatter(i[0], i[1], c = 'blue')
        elif (kmeans_model.labels_[number] == 2):
            plt.scatter(i[0], i[1], c = 'yellow')
        elif (kmeans_model.labels_[number] == 3):
            plt.scatter(i[0], i[1], c = 'brown')


    for i, t in enumerate(test_data):
        if (cluster_index[i] == 0):
            plt.scatter(t[0], t[1], c = 'red')
        elif (cluster_index[i] == 1):
            plt.scatter(t[0], t[1], c = 'blue')
        elif (cluster_index[i] == 2):
            plt.scatter(t[0], t[1], c = 'yellow')
        elif (cluster_index[i] == 3):
            plt.scatter(t[0], t[1], c = 'brown')


    plt.plot(final_centroid[:, 0], final_centroid[:, 1], 'rD', markersize = 12, label = 'final_centroid')
    # X축은 각성도, Y축은 긍정도 값으로 라벨 붙이기 
    plt.xlabel('Arousal')
    plt.ylabel('Valence')
    plt.legend()
    plt.show()
    plt.close()


def TestResult(final_centroids, test_data):
    print("TestResult")
    # 테스트 데이터와 각 클러스터의 중심값의 거리값을 구하여 디스턴스 식별자에 저장
    # 클러스터 인덱스에 테스트 데이터를 별도 가장 짧은 거리값에 인덱스를 저장
    # 최종적으로 클러스터 인덱스를 리턴하면 테스트 데이터가 어떤 클러스터에 속하는지 확인

if __name__ == '__main__':
    main()