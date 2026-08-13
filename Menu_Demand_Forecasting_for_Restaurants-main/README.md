# 식음업장 메뉴 수요 예측 프로젝트 (Sales Volume Prediction Project)
https://dacon.io/competitions/official/236559/overview/description
---

## 프로젝트 개요
이 프로젝트는 영업장명명로부터 수집한 매출 데이터를 바탕으로, **주말 과 월별 패턴을 해석하고 고객 수요 예측 AI를 구축**하기 위한 데이터 분석 및 학습 노드입니다.

> **목적:**  
> - 월과 요일 별 매출 패턴 보상  
> - 메뉴/매장 별 고객 수요 구조 해석  
> - 그리고 수요 예측 모델 구축 및 평가  

---

## 1. 데이터 기본 확인 및 결측치 처리

- **수치형 결측치:** 평균으로 대체
- **범주형 결측치:** 최빈값(mode)으로 대체

또한 `영업일자` 기준으로 `요일` 필드를 추가하여 시각적 보고에 활용하였습니다.

```python
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
```

---

## 2. EDA (감지적 데이터 분석)

### (1) 요일별 평균 매출수량
<img width="840" height="544" alt="image" src="https://github.com/user-attachments/assets/9de2db4a-0295-49c1-958e-cb6a5d2dbd77" />

> 주중보다 주말(토, 일)에 매출이 보상적으로 증가하여 여건 월별 수요 패턴을 보여줍니다.

### (2) 월별 매출수량 추이
<img width="873" height="544" alt="image" src="https://github.com/user-attachments/assets/991b02b9-3d23-4384-a45e-689d56dd334f" />
> 1 ~ 2월은 최고 수준을 보였으며, 이후 3월에 감소하였습니다. 연초 특정 시점에 월간 소비 집중이 그렇게 되었을 것으로 해석됩니다.

### (3) 메뉴별 평균 매출수량 TOP10
<img width="1159" height="544" alt="image" src="https://github.com/user-attachments/assets/9a542c4e-2df3-4977-9d51-c2e6798a9cb5" />

> `포레스트릿_꼬지어묵`, `화담술주말_해물파전`, `포레스트릿_떡볶이`가 가장 강한 보유를 보였으며, 고정 고객구의 대표적 메뉴로 보여줍니다.

### (4) 업체별 매출수량 분포
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/c2fddc7e-61b9-4f9a-94f7-376ad00e7cda" />


> `포레스트릿`, `카페테리아`, `화담술주말`이 강한 방면을 보여주며, 직접적인 고객 수요가 확인되는 것을 발견할 수 있습니다.
 
---

## 3. KMeans 군집화 및 패삭변수 생성

3개 군집으로 분류한 결과, `cluster 1`이 매출수량에 가장 그룹한 영향을 미치며, 특정 메뉴그룹이 전체 매출을 이보하는 구조를 보여줍니다.

---

## 4. GLM 형식을 통한 월/요일 별 인과 분석
<img width="628" height="394" alt="image" src="https://github.com/user-attachments/assets/5d7a0033-7344-4ae4-9897-c10211787935" />


> **결과:**  
> - 토요일: 회귀계수 **+1.6013**, 유의합
> - 화~목요일: 음의 효과 가지며 평일 매출 저소
> - `cluster 1`: 계수 **+332.74** 로 강한 양의 효과

> 이 결과는 주말에 고객 수요가 집중되고 특정 메뉴/면장 그룹이 전체 매출을 가지고 있을 것을 보여줍니다.

---

## 5. 회귀 모델 성능 비교

| 모델 | MAE | RMSE | R² |
|:--|:--|:--|:--|
| Ridge | 14.38 | 35.72 | 0.213 |
| RandomForest | **5.97** | **19.19** | **0.773** |
| GradientBoosting | 7.90 | 20.21 | 0.748 |

> **RandomForest 모델이 가장 높은 R² (0.773)과 최소 MAE(5.97)를 보여 가장 효과적입니다.**  
> Ridge 모델은 선형성 구조에서 설명력이 저했습니다.

---

## 6. LightGBM 기반 Stacking

- **Base Models:** RandomForest, GradientBoosting  
- **Meta Model:** LightGBM  
- **Feature Reduction:** 중요도 하위 30% 제거 (132개 주요 변수만 사용)

| Metric | Value |
|:--|:--|
| MAE | 6.30 |
| RMSE | 18.74 |
| R² | **0.7836** |

> LightGBM이 메타모델로 동작하며 과적합을 줄이고 비선형적 패턴을 정교하게 포착함.  
> Feature selection 이후에도 성능이 향상되어 모델 안정성과 해석력을 동시에 확보함.

---

## 7. Feature Importance 분석
<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/9131293e-b906-43a4-ab74-f8572971a05f" />


> - **‘월’** 변수의 중요도가 가장 높아, 계절성과 프로모션 등 시간 요인이 핵심.  
> - **‘cluster’** 및 **‘요일_Saturday’, ‘요일_Sunday’** 역시 상위권 → 주말 중심 매출 집중 구조.  
> - **‘메뉴명_고치어묵’, ‘영업장명_화담숲주막’** 등 주요 매장 및 메뉴 관련 변수도 높은 영향력을 보임.

> **종합적으로**, 시간(월·요일) 요인 + 구조적 요인(cluster) + 개별 수요 특성(메뉴·매장)을 균형 있게 반영하여 높은 설명력을 확보하였으며, 이는 향후 수요 예측 및 재고 관리 전략 수립에 유용한 인사이트를 제공합니다.

---

## 종합 결론
- **주말(특히 토요일)**과 **연초(1~2월)**의 매출 집중 현상 확인  
- **특정 메뉴군(고치어묵·해물파전·떡볶이)**이 핵심 매출을 견인  
- **RandomForest + LightGBM 스태킹 모델**이 최고 성능(R² = 0.7836)을 기록  
- **Feature Importance 분석**을 통해, 시간·메뉴·매장 요인이 주요 예측 인자로 작용함을 확인  

---

**Author**  
**윤해정 (Yoon Haejeong)**  
heajeongy@naver.com  
Data Analytics & Visualization | Python, SQL, Power BI  

