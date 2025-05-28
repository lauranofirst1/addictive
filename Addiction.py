import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1) 데이터 불러오기
df = pd.read_csv("Students Social Media Addiction.csv")

# 2) 사용할 수치형 컬럼
numeric_cols = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score'
]

# 3) 수치형 데이터만 추출 및 결측치 제거
data_numeric = df[numeric_cols].dropna()

# 4) 표준화
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# 5) KMeans 클러스터링 (3개)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# 6) 원본 데이터에 클러스터 정보 병합
df.loc[data_numeric.index, 'Cluster'] = clusters

# 7) 클러스터 → 상태 레이블로 변환
cluster_labels = {0: 'Normal', 1: 'Addicted', 2: 'Moderate'}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

# 8) 수치형 변수 평균 (클러스터 상태별)
cluster_profiles = df.groupby('Cluster_Label')[numeric_cols].mean().round(2)
print("=== 상태별 수치형 변수 평균 ===")
print(cluster_profiles)

# 9) 관계 상태 필터링
rel_filtered = df.loc[data_numeric.index, :].copy()
rel_filtered = rel_filtered[rel_filtered['Relationship_Status'].isin(['Single', 'In Relationship'])].copy()

# 10) 나이 + 학업 단계 복합 그룹 생성
rel_filtered['Age_Edu_Group'] = (
    rel_filtered['Age'].astype(int).astype(str)
    + '_' + rel_filtered['Academic_Level']
)

# 11) 복합 그룹 기준 상태 분포
ageedu_cluster_dist = pd.crosstab(
    rel_filtered['Age_Edu_Group'],
    rel_filtered['Cluster_Label'],
    normalize='index'
)
print("\n=== 복합 그룹(Age + Academic_Level) 기준 상태 분포 (비율) ===")
print(ageedu_cluster_dist.round(3))

# 12) 복합 그룹별 주요 상태 해석
print("\n=== 복합 그룹별 주요 상태 해석 ===")
for group in ageedu_cluster_dist.index:
    dominant = ageedu_cluster_dist.loc[group].idxmax()
    percentage = ageedu_cluster_dist.loc[group].max()
    print(f"{group} 그룹은 주로 '{dominant}' 상태에 속함 ({percentage:.2%})")

# 13) 관계 상태별 중독 점수 평균 (복합 그룹 기준)
add_score_ageedu_rel = rel_filtered.groupby(
    ['Age_Edu_Group', 'Relationship_Status']
)['Addicted_Score'].mean().unstack()

print("\n=== 복합 그룹(Age + Academic_Level) 기준 관계 상태별 중독 점수 평균 ===")
print(add_score_ageedu_rel.round(2))

# 14) 시각화 - 상태 분포
plt.figure(figsize=(10,6))
ageedu_cluster_dist.plot(
    kind='bar', stacked=True, colormap='tab20', figsize=(10,6)
)
plt.title('Addiction Status Distribution by Age_Edu_Group')
plt.xlabel('Age_Edu_Group')
plt.ylabel('Proportion')
plt.legend(title='Addiction Status')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 15) 시각화 - 중독 점수 평균
plt.figure(figsize=(10,6))
add_score_ageedu_rel.plot(
    kind='bar', colormap='Set2', figsize=(10,6)
)
plt.title('Average Addiction Score by Age_Edu_Group and Relationship_Status')
plt.xlabel('Age_Edu_Group')
plt.ylabel('Average Addiction Score')
plt.legend(title='Relationship Status')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

