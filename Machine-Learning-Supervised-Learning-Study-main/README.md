# Machine Learning Supervised Learning Study

다양한 데이터셋을 활용하여 지도학습(Supervised Learning)의 전 과정을 실습하고 시각화 해본 자료입니다.  
Scikit-learn, XGBoost, LightGBM, CatBoost, PyCaret, AutoGluon 등 주요 머신러닝 라이브러리를 포함하여 학습하였습니다.

---
**목차**

## 1. 데이터 전처리 (Data Preprocessing)
- 결측치 처리 (dropna, fillna)  
- Label Encoding 및 One-Hot Encoding  
- 스케일링 기법 비교  
  - StandardScaler (정규분포형 데이터)  
  - MinMaxScaler (0~1 범위)  
  - RobustScaler (이상치 대응)  
- 피처 선택 및 생성  
  - 다중공선성(VIF)  
  - 상관계수 히트맵 (matshow, corr)  
  - 시계열 분해 (year, month, week, weekday 등)

---

## 2. 회귀 모델 (Regression Models)
### 기본 모델
- Linear Regression  
- SGD Regressor (확률적 경사하강법)  
- Polynomial Regression (비선형 확장)

### 규제 기반 모델
- Ridge (L2 Regularization)  
- Lasso (L1 Regularization)  
- ElasticNet (L1 + L2 혼합)

### 고급 회귀 모델
- DecisionTreeRegressor  
- RandomForestRegressor  
- SVR (Linear / Polynomial / RBF Kernel)  
- GradientBoostingRegressor  
- XGBoost / LightGBM / CatBoost

### 평가 지표
- MAE / MSE / RMSE  
- R² Score  
- 교차검증 (K-Fold)

---

## 3. 분류 모델 (Classification Models)
### 기본 모델
- Logistic Regression  
- SVM (Linear / Polynomial / RBF Kernel)

### 트리 기반 모델
- DecisionTreeClassifier  
- RandomForestClassifier  
- GradientBoostingClassifier  
- AdaBoostClassifier

### 불균형 데이터 처리
- SMOTE / SMOTENC 오버샘플링  
- Stratified K-Fold  
- Class Weight 조정  

### 평가 지표
- Accuracy / Precision / Recall / F1 / ROC-AUC  
- Confusion Matrix 시각화  

---

## 4. 앙상블 학습 (Ensemble Learning)
- Voting Classifier (Hard / Soft)  
- Bagging (Bootstrap Aggregation)  
- Boosting (AdaBoost, GradientBoost, XGBoost, LightGBM, CatBoost)  
- Stacking (다중 모델 메타학습)

---

## 5. AutoML 및 하이퍼파라미터 최적화 (AutoML & Optimization)
- PyCaret : 자동 모델 비교 및 성능 평가  
- Optuna : Bayesian 기반 하이퍼파라미터 탐색  
- Bayesian Optimization : XGBoost 성능 튜닝  
- AutoGluon : TabularPredictor를 통한 자동 모델 학습 및 평가  

---

## 6. 시각화 및 해석 (Visualization & Interpretation)
- Feature Importance Barplot  
- Decision Tree Graphviz 시각화  
- Scatter, Line, Bar, Pie Chart  
- 예측값 vs 실제값 비교 그래프  

---

## 7. 프로젝트별 예시 (Case Studies)
### Jeju Special Product Price Prediction (Dacon)
- K-Fold 교차검증  
- CatBoostRegressor  
- Feature Importance 시각화  
- RMSE 기반 성능 비교  

### Titanic Survival Prediction
- SMOTE + DecisionTreeClassifier  
- Feature Importance 도출  

### Credit Score Classification
- SMOTENC + VotingClassifier  
- CatBoost + LGBM + RandomForest 앙상블  

---
