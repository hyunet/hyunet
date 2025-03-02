import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기 (실제 당뇨병 데이터가 없으므로 샘플 데이터 사용)
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = (data.target > 150).astype(int)  # 고혈당 기준으로 임의 라벨링

# 데이터 표준화
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('target', axis=1))
y = df['target']

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 웹 앱 인터페이스
st.title("당뇨병 환자 예측 웹사이트")
st.write("학생들이 입력한 건강 속성을 바탕으로 당뇨병 가능성을 예측합니다.")

# 사용자 입력 받기
def user_input():
    age = st.slider("나이", 20, 80, 40)
    bmi = st.number_input("BMI 지수", min_value=10.0, max_value=50.0, value=25.0)
    bp = st.number_input("혈압 (mmHg)", min_value=60.0, max_value=200.0, value=120.0)
    s1 = st.number_input("콜레스테롤 수치", min_value=100.0, max_value=300.0, value=200.0)
    s2 = st.number_input("혈당 수치", min_value=60.0, max_value=300.0, value=100.0)
    data = np.array([[age, bmi, bp, s1, s2, 0, 0, 0, 0, 0]])
    return data

user_data = user_input()
user_data = scaler.transform(user_data)

if st.button("예측하기"):
    prediction = model.predict(user_data)
    if prediction[0] == 1:
        st.error("⚠️ 당뇨병 위험이 높습니다. 건강 관리를 권장합니다.")
    else:
        st.success("✅ 당뇨병 위험이 낮습니다. 앞으로도 건강을 유지하세요.")

st.write("2025.03.02")

