#라이브러리 넣기
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일에서 데이터 읽어오기
data = pd.read_csv('MR\mosquito_Indicator.csv')

# 날짜 데이터 문자열을 숫자로 변환
data['date'] = pd.to_datetime(data['date'], format="%m/%d")
data['date'] = data['date'].apply(lambda x: x.toordinal())

# 필요한 특성과 타겟 변수 선택
features = ['date', 'rain(mm)', 'mean_T', 'min_T', 'max_T']
target = 'mosquito_Indicator'
X = data[features]
y = data[target]

# 훈련 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

# 하이퍼파라미터 튜닝을 위한 그리드 서치 설정
param_grid = {
    'n_estimators': [50, 100, 200],  # 트리의 개수
    'max_depth': [None, 5, 10],  # 트리의 최대 깊이
    'min_samples_split': [2, 5, 10],  # 내부 노드를 분할하기 위한 최소 샘플 수
    'min_samples_leaf': [1, 2, 4]  # 리프 노드의 최소 샘플 수
}

# 랜덤 포레스트 모델 생성
model = RandomForestRegressor(random_state=24, n_jobs=-1, oob_score=True)

# 그리드 서치를 사용한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(model, param_grid, cv=8)
grid_search.fit(X_train, y_train)

# 최적의 모델 및 하이퍼파라미터 출력
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Model:", best_model)
print("Best Parameters:", best_params)

# 학습된 최적의 모델로 예측 수행
y_pred = best_model.predict(X_test)

# R2 스코어 계산
r2 = r2_score(y_test, y_pred)
print("테스트 세트 R2 스코어:", r2)

new_data = np.array([[pd.to_datetime('7/2', format="%m/%d").toordinal(), 16.5, 21.1, 16.5, 28.4], 
                     [pd.to_datetime('8/10', format="%m/%d").toordinal(), 13.5, 14.5, 12.5, 18.1]])  # 새로운 데이터 예시 (날짜를 숫자로 변환)


# 학습된 최적의 모델로 예측 수행
predictions = best_model.predict(new_data)

print("새로운 데이터에 대한 예측:")
print(predictions)

# 최적 모델의 훈련 및 테스트 점수 계산
score_train = best_model.score(X_train, y_train)
print("훈련 점수:", score_train)
score_test = best_model.score(X_test, y_test)
print("테스트 점수:", score_test)

# 특성 중요도 출력
print("특성 중요도:", best_model.feature_importances_)

# 모델 저장
#joblib.dump(best_model, 'random_forest_model.pkl')

# 데이터의 산점도 행렬
sns.pairplot(data[features])
plt.show()

# 각 특성의 분포
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
for i, feature in enumerate(features):
    ax = axes[i//3, i%3]
    sns.histplot(data=data, x=feature, ax=ax)
    ax.set_xlabel(feature)
plt.tight_layout()
plt.show()