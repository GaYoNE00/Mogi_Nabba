from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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

# 랜덤 포레스트 모델 생성 및 학습
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 새로운 데이터 예측
new_data = np.array([[pd.to_datetime('7/2', format="%m/%d").toordinal(), 16.5, 21.1, 16.5, 28.4], 
                     [pd.to_datetime('8/10', format="%m/%d").toordinal(), 13.5, 14.5, 12.5, 18.1]])  # 새로운 데이터 예시 (날짜를 숫자로 변환)
predictions = model.predict(new_data)


print("새로운 데이터에 대한 예측:")
print(predictions)

score = model.score(X_train, y_train)
print("훈련  점수", score)
score = model.score(X_test, y_test)
print("테스트  점수", score)

print("특성 중요도 : ", model.feature_importances_)