import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def train_xgb(X_train, y_train, model_path='demo/xgb_model.joblib', random_state=None):

    xgb_model = xgb.XGBClassifier(random_state=random_state)

    xgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, model_path)
    print(f"Model saved to {model_path}")

    return xgb_model


def infer_xgb(X_test, model_path='demo/xgb_model.joblib'):
    xgb_model = joblib.load(model_path)

    y_pred = xgb_model.predict(X_test)

    return y_pred
