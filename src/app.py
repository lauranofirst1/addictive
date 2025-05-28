import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 스타일 설정
st.set_page_config(page_title="SNS 중독 분석 시스템", layout="wide")

# 데이터 불러오기
df = pd.read_csv(os.path.join("..", "data", "Students_Social_Media_Addiction.csv"))

st.title("📱 SNS 중독 분석 및 예측 시스템")

tab1, tab2 = st.tabs(["📊 전체 데이터 분석", "🧠 예측 + 코칭"])

# ================== 📊 전체 분석 탭 ==================
with tab1:
    st.header("🔍 클러스터링 기반 중독 상태 분석")

    numeric_cols = [
        'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
        'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score'
    ]

    data_numeric = df[numeric_cols].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_numeric)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df.loc[data_numeric.index, 'Cluster'] = clusters
    cluster_labels = {0: 'Normal', 1: 'Addicted', 2: 'Moderate'}
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

    cluster_profiles = df.groupby('Cluster_Label')[numeric_cols].mean().round(2)
    st.subheader("📌 상태별 수치형 변수 평균")
    st.dataframe(cluster_profiles)

    # 복합 그룹 생성 및 분석
    rel_filtered = df.loc[data_numeric.index, :].copy()
    rel_filtered = rel_filtered[rel_filtered['Relationship_Status'].isin(['Single', 'In Relationship'])].copy()
    rel_filtered['Age_Edu_Group'] = (
        rel_filtered['Age'].astype(int).astype(str) + '_' + rel_filtered['Academic_Level']
    )

    ageedu_cluster_dist = pd.crosstab(
        rel_filtered['Age_Edu_Group'],
        rel_filtered['Cluster_Label'],
        normalize='index'
    )

    st.subheader("📊 복합 그룹별 상태 비율 분포")
    st.dataframe(ageedu_cluster_dist.round(3))

    st.subheader("📈 시각화: 상태 분포")
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ageedu_cluster_dist.plot(kind='bar', stacked=True, colormap='tab20', ax=ax1)
    plt.title('Addiction Status Distribution by Age_Edu_Group')
    plt.xlabel('Age_Edu_Group')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig1)

    # 관계 상태별 점수 평균
    add_score_ageedu_rel = rel_filtered.groupby(
        ['Age_Edu_Group', 'Relationship_Status']
    )['Addicted_Score'].mean().unstack()

    st.subheader("📈 시각화: 중독 점수 평균 (연애 여부)")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    add_score_ageedu_rel.plot(kind='bar', colormap='Set2', ax=ax2)
    plt.title('Average Addiction Score by Age_Edu_Group and Relationship_Status')
    plt.xlabel('Age_Edu_Group')
    plt.ylabel('Average Addiction Score')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig2)

# ================== 🧠 예측 + 코칭 탭 ==================
with tab2:
    st.header("🧠 SNS 중독 상태 예측 + 코칭 피드백")

    # 모델 불러오기
    MODEL_DIR = os.path.join("..", "models")
    clf = joblib.load(os.path.join(MODEL_DIR, "addiction_classifier.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "addiction_regressor.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    st.markdown("아래 항목을 입력하면 당신의 중독 상태를 예측하고 개선 방법을 제안합니다.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("나이 (Age)", 10, 30, 20)
        sleep = st.slider("수면 시간 (시간)", 0.0, 12.0, 6.0, 0.5)
    with col2:
        usage = st.slider("SNS 사용 시간 (시간)", 0.0, 10.0, 3.0, 0.5)
        mental = st.slider("정신 건강 점수 (0~100)", 0, 100, 50)

    if st.button("📈 중독 상태 예측 + 피드백 받기"):
        input_df = pd.DataFrame([[age, usage, sleep, mental]],
                                columns=['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score'])

        pred_class = clf.predict(input_df)[0]
        label = le.inverse_transform([pred_class])[0]
        base_score = reg.predict(input_df)[0]
        usage_score = reg.predict([[age, usage - 1, sleep, mental]])[0] if usage > 1 else base_score
        sleep_score = reg.predict([[age, usage, sleep + 1, mental]])[0] if sleep < 11 else base_score

        st.success(f"🧠 예측된 중독 상태: **{label}**")
        st.metric("예측된 중독 점수", f"{base_score:.1f}")

        st.subheader("💡 개선 코칭 피드백")
        if base_score - usage_score > 5:
            st.write(f"📉 하루 사용시간을 1시간 줄이면 중독 점수가 약 **{base_score - usage_score:.1f}점 감소**할 수 있어요.")
        if sleep_score - base_score < -5:
            st.write(f"😴 수면시간을 1시간 늘리면 중독 점수가 약 **{base_score - sleep_score:.1f}점 감소**할 수 있어요.")
        if base_score - usage_score <= 5 and sleep_score - base_score >= -5:
            st.write("✅ 현재 상태는 안정적입니다. 지금처럼 규칙적인 생활을 유지하세요!")
