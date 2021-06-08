import shap


def get(model, X_data):

    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_data)
