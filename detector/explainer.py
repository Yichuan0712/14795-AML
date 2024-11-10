"""
SHAP
Feature Importance
TreeExplainer
LIME
PDP

import matplotlib.pyplot as plt
from xgboost import plot_importance
plot_importance(xgb_model)
plt.show()

from sklearn.inspection import plot_partial_dependence
plot_partial_dependence(xgb_model, X_train, [0, (0, 1)], grid_resolution=50)
plt.show()

"""