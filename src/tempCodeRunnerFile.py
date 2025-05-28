 # 모델 불러오기
    MODEL_DIR = os.path.join("..", "models")
    clf = joblib.load(os.path.join(MODEL_DIR, "addiction_classifier.pkl"))
    reg = joblib.load(os.path.join(MODEL_DIR, "addiction_regressor.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))