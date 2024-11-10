import joblib
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular
from xgboost import plot_importance
from sklearn.inspection import plot_partial_dependence


def explain_xgb(X_train, X_test, model_path='demo/xgb_model.joblib'):
    xgb_model = joblib.load(model_path)

    # 1. SHAP 解释器
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # 绘制 SHAP 总结图
    print("SHAP Summary Plot:")
    shap.summary_plot(shap_values, X_test)

    # 2. 特征重要性图（XGBoost 自带）
    print("Feature Importance Plot:")
    plot_importance(xgb_model)
    plt.show()

    # 3. LIME 解释器（适用于单个数据点）
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['Class 0', 'Class 1'],
        mode='classification'
    )
    # 解释第一个样本
    exp = lime_explainer.explain_instance(X_test.iloc[0], xgb_model.predict_proba)
    print("LIME Explanation for the first test instance:")
    exp.show_in_notebook(show_all=False)

    # 4. 部分依赖图（PDP）
    print("Partial Dependence Plot:")
    plot_partial_dependence(xgb_model, X_train, [0, (0, 1)], grid_resolution=50)
    plt.show()
