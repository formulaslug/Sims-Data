import joblib
import json
import numpy as np

explainer = joblib.load('explainer.joblib')

summary = {}

# Model info
model = explainer.model
summary['model_type'] = type(model).__name__
try:
    params = model.get_params()
    summary['model_params'] = {k: str(v) for k, v in params.items()}
except:
    summary['model_params'] = 'unavailable'

# Feature names
try:
    summary['feature_names'] = list(explainer.columns)
except:
    summary['feature_names'] = 'unavailable'

# Feature importances
try:
    importances = explainer.importances
    if hasattr(importances, 'to_dict'):
        summary['feature_importances'] = importances.head(20).to_dict()
    else:
        summary['feature_importances'] = str(importances)
except:
    summary['feature_importances'] = 'unavailable'

# Class/label info
try:
    summary['classes'] = list(explainer.labels)
except:
    try:
        summary['classes'] = list(explainer.y.unique())
    except:
        summary['classes'] = 'unavailable'

# Prediction distribution
try:
    preds = explainer.preds
    summary['pred_distribution'] = {
        'mean': float(np.mean(preds)),
        'min': float(np.min(preds)),
        'max': float(np.max(preds)),
        'pct_positive': float(np.mean(preds > 0.5))
    }
except:
    summary['pred_distribution'] = 'unavailable'

# Label distribution
try:
    y = explainer.y
    vals, counts = np.unique(y, return_counts=True)
    summary['label_distribution'] = {str(int(v)): int(c) for v, c in zip(vals, counts)}
except:
    summary['label_distribution'] = 'unavailable'

# SHAP summary if available
try:
    shap_vals = explainer.shap_values_df
    summary['top_shap_features'] = shap_vals.abs().mean().sort_values(ascending=False).head(15).to_dict()
except:
    summary['top_shap_features'] = 'unavailable'

print(json.dumps(summary, indent=2, default=str))