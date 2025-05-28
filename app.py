# app.py
import streamlit as st
import pandas as pd
import joblib

# 모델 불러오기
clf = joblib.load("addiction_classifier.pkl")
reg = joblib.load("addiction_regressor.pkl")
le = joblib.load("label_encoder.pkl")

st.title("📊 SNS 중독 상태 예측기 + 📱 개선 코칭 시스템")
st.markdown("사용자의 입력 정보를 바탕으로 SNS 중독 상태를 예측하고, 개선을 위한 코칭을 제공합니다.")

# 사용자 입력
age = st.slider("나이 (Age)", 10, 30, 20)
usage = st.slider("하루 평균 SNS 사용시간 (시간)", 0.0, 10.0, 3.0, 0.5)
sleep = st.slider("하루 평균 수면시간 (시간)", 0.0, 12.0, 6.0, 0.5)
mental = st.slider("정신 건강 점수 (0~100)", 0, 100, 50)

if st.button("📈 중독 상태 예측 + 코칭 받기"):
    input_data = pd.DataFrame([[age, usage, sleep, mental]], columns=['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score'])

    # 중독 상태 예측
    pred_class = clf.predict(input_data)[0]
    label = le.inverse_transform([pred_class])[0]

    # 중독 점수 예측
    base_score = reg.predict(input_data)[0]
    usage_score = reg.predict([[age, usage - 1, sleep, mental]])[0] if usage > 1 else base_score
    sleep_score = reg.predict([[age, usage, sleep + 1, mental]])[0] if sleep < 11 else base_score

    st.subheader(f"🧠 예측된 중독 상태: **{label}**")
    st.metric("예측 중독 점수", f"{base_score:.1f}")

    st.subheader("💡 개선 코칭 피드백")
    if base_score - usage_score > 5:
        st.write(f"📉 하루 사용시간을 1시간 줄이면 중독 점수가 약 **{base_score - usage_score:.1f}점 감소**할 수 있어요.")
    if sleep_score - base_score < -5:
        st.write(f"😴 수면시간을 1시간 늘리면 중독 점수가 약 **{base_score - sleep_score:.1f}점 감소**할 수 있어요.")
    if base_score - usage_score <= 5 and sleep_score - base_score >= -5:
        st.write("✅ 현재 상태는 안정적입니다. 지금처럼 규칙적인 생활을 유지하세요!")
