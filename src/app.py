import streamlit as st
import pandas as pd
import joblib

# ====== 1. 웹 페이지 기본 설정 ======
st.set_page_config(page_title="SNS 중독 예측 시스템", layout="centered")
st.title("📱 SNS 중독 예측 + 코칭 피드백 시스템")

st.markdown("""
사용자 정보를 입력하면 SNS 중독 점수를 예측하고, 개선 방향에 대한 피드백을 제공합니다.
""")

# ====== 2. 사용자 입력 UI 구성 (Age, Academic_Level 제외) ======
usage = st.slider("하루 평균 SNS 사용 시간 (시간)", 0.0, 10.0, 3.0, step=0.5)
sleep = st.slider("하루 수면 시간 (시간)", 0.0, 12.0, 6.0, step=0.5)
mental = st.slider("정신 건강 점수 (0~10)", 0, 10, 5)
conflicts = st.slider("SNS 관련 갈등 수준 (0~10)", 0, 10, 2)

gender = st.selectbox("성별", ["Male", "Female"])
country = st.selectbox("국가", ["India", "USA", "UK", "Canada", "Bangladesh"])
platform = st.selectbox("주로 사용하는 SNS 플랫폼", ["Instagram", "Facebook", "TikTok", "Twitter", "YouTube"])
relationship = st.selectbox("연애 상태", ["Single", "In Relationship", "Complicated"])
affects_academic = st.selectbox("SNS가 학업에 영향을 미치나요?", ["Yes", "No"])

# ====== 3. 예측 수행 버튼 ======
if st.button("📊 예측 수행"):
    # ====== 4. 입력값 구성 (Age, Academic_Level 제외) ======
    input_data = pd.DataFrame([{
        "Avg_Daily_Usage_Hours": usage,
        "Sleep_Hours_Per_Night": sleep,
        "Mental_Health_Score": mental,
        "Conflicts_Over_Social_Media": conflicts,
        "Gender": gender,
        "Country": country,
        "Most_Used_Platform": platform,
        "Relationship_Status": relationship,
        "Affects_Academic_Performance": affects_academic
    }])

    # ====== 5. 모델 로드 ======
    model = joblib.load("sns_addiction_model_2.pkl")

    # ====== 6. 예측 수행 ======
    predicted_score = model.predict(input_data)[0]

    # ====== 7. 중독 상태 분류 ======
    def categorize(score):
        if score <= 5.0:
            return "정상 ✅"
        elif score <= 7.0:
            return "보통 ⚠️"
        elif score <= 8.5:
            return "나쁨 ❌"
        else:
            return "매우 나쁨 🚨"

    status = categorize(predicted_score)

    # ====== 8. 출력 ======
    st.subheader("📈 예측 결과")
    st.metric("예측된 중독 점수", f"{predicted_score:.2f}")
    st.success(f"중독 상태: **{status}**")

    # ====== 9. 피드백 코칭 ======
    st.subheader("💡 개선 피드백")
    if predicted_score > 5:
        if usage > 2:
            st.write("📉 하루 SNS 사용 시간을 **1시간 줄이면**, 중독 점수가 **낮아질 수 있어요.**")
        if sleep < 7:
            st.write("😴 수면 시간을 늘리면 중독 점수를 낮추는 데 도움이 됩니다.")
        if mental < 6:
            st.write("🧘 정신 건강 관리가 필요해요. 명상이나 운동을 추천합니다.")
    else:
        st.write("🎉 현재 상태는 안정적이에요. 계속 좋은 습관을 유지해 주세요!")
