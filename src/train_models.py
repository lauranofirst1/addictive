import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Download latest version
path = kagglehub.dataset_download("adilshamim8/social-media-addiction-vs-relationships")
print("Path to dataset files:", path)

csv_path = os.path.join(path, "Students_Social_Media_Addiction.csv")
df = pd.read_csv(csv_path)

# 수치형 컬럼
numeric_cols = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score'
]

data_numeric = df[numeric_cols].dropna()

# 클러스터링
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df.loc[data_numeric.index, 'Cluster'] = clusters
cluster_labels = {0: 'Normal', 1: 'Addicted', 2: 'Moderate'}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

# 예측 모델 학습
features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']
target_class = 'Cluster_Label'
target_reg = 'Addicted_Score'

df_model = df[features + [target_class, target_reg]].dropna()

# 분류 모델
X_class = df_model[features]
y_class = df_model[target_class]
le = LabelEncoder()
y_class_encoded = le.fit_transform(y_class)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_class, y_class_encoded)

# 회귀 모델
X_reg = df_model[features]
y_reg = df_model[target_reg]
reg = LinearRegression()
reg.fit(X_reg, y_reg)

# 저장
joblib.dump(clf, "addiction_classifier.pkl")
joblib.dump(reg, "addiction_regressor.pkl")
joblib.dump(le, "label_encoder.pkl")
print("모델 저장 완료.")
