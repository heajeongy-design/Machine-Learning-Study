# Unsupervised Learning & Feature Engineering Study

본 프로젝트는 **비지도학습(Unsupervised Learning)** 기반의 대표 기법인 **PCA(주성분 분석)**, **K-Means 클러스터링**,  
그리고 **군집 기반 파생변수 생성**을 통해 데이터의 구조적 패턴을 시각화하고, 지도학습 모델의 성능 향상을 탐구한 실습 하였습니다.

---

## 1. Overview
- PCA를 활용한 차원 축소 및 시각화  
- K-Means, K-Means++를 통한 군집화  
- Elbow / Silhouette 기법으로 최적의 K 탐색  
- KMeans 기반 파생변수를 활용한 Decision Tree 모델 성능 비교  
- 데이터: Iris / Breast Cancer / Titanic

---

## 2. 주요 라이브러리 (Libraries)
- pandas, numpy, matplotlib, seaborn, plotly  
- scikit-learn (PCA, KMeans, silhouette_score, DecisionTreeClassifier)  
- load_iris, load_breast_cancer, train_test_split, f1_score 등  

---

## 3. PCA (Principal Component Analysis)
- 4차원 데이터를 2차원으로 축소  
- 각 품종(setosa, versicolor, virginica)을 색상으로 구분  
- 주성분 간 분포를 시각적으로 비교  

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
