import shap
import numpy as np
import pandas as pd



def analyze_with_shap(model, X_train, sample):
    if isinstance(sample, pd.Series):
        sample = sample.to_frame().T
    
    predicted_class = int(model.predict(sample)[0])
    model_type = str(type(model)).lower()
    feature_names = list(X_train.columns)
    
    try:
        if any(t in model_type for t in ["forest", "tree", "boost", "gbm", "catboost"]):
            explainer = shap.TreeExplainer(model)
            raw = explainer.shap_values(sample)
        elif any(t in model_type for t in ["keras", "tensorflow", "torch"]):
            background = shap.sample(X_train, min(100, len(X_train))).values
            explainer = shap.DeepExplainer(model, background)
            raw = explainer.shap_values(sample.values)
        else:
            explainer = shap.Explainer(model, X_train)
            res = explainer(sample)
            raw = res.values if hasattr(res, "values") else res
    except Exception:
        background = shap.sample(X_train, min(50, len(X_train)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        raw = explainer.shap_values(sample)

    if isinstance(raw, list):
        vals = raw[predicted_class]
    else:
        vals = np.array(raw)

    if vals.ndim == 3:         
        vals = vals[0, :, predicted_class]
    elif vals.ndim == 2:         
        vals = vals[0]
    
    shap_dict = {name: float(val) for name, val in zip(feature_names, vals)}
    sample_dict = {name: float(sample[name].iloc[0]) for name in feature_names}

    return shap_dict, sample_dict, predicted_class