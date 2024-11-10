import joblib
import shap
import matplotlib.pyplot as plt
# from lime import lime_tabular
from xgboost import plot_importance
# from sklearn.inspection import plot_partial_dependence

def explain_xgb_demo(X_test, model_path='demo/xgb_model.joblib'):
    xgb_model = joblib.load(model_path)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    print("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test)

    print("Feature Importance Plot:")
    plot_importance(xgb_model)
    plt.show()

# explain_xgb_demo(X, model_path='/content/14795-AML/demo/xgb_model1_demo.joblib')

import pandas as pd
import numpy as np


def calculate_amount_correlation(csv_path, X_test):
    df = pd.read_csv(csv_path)

    amount_data = df['Amount'].values

    X_test_df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])

    X_test_df['Amount'] = amount_data[:X_test_df.shape[0]]

    correlation_matrix = X_test_df.corr()

    amount_correlation = correlation_matrix['Amount'].drop('Amount').sort_values(ascending=False)

    return amount_correlation


# calculate_amount_correlation('/content/14795-AML/demo/test0.02.csv', X)

