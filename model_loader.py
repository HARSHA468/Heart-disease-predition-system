"""
Model Loader for Heart Disease Prediction
Ensures models are loaded only once and shared across modules.
"""
import joblib
from config import XGBOOST_MODEL_PATH, ECG_MODEL_PATH

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[WARNING] TensorFlow not installed. ECG model will be unavailable.")
    print("[INFO] TensorFlow does not yet support Python 3.14. Use Python 3.11 for full functionality.")

_xgb_model = None
_ecg_model = None

def load_models():
    """Load models into memory. Should be called once on app startup."""
    global _xgb_model, _ecg_model
    
    if _xgb_model is None:
        print("Loading XGBoost model from:", XGBOOST_MODEL_PATH)
        _xgb_model = joblib.load(XGBOOST_MODEL_PATH)
        
    if _ecg_model is None:
        if TENSORFLOW_AVAILABLE:
            print("Loading ECG CNN model from:", ECG_MODEL_PATH)
            _ecg_model = load_model(ECG_MODEL_PATH)
        else:
            print("[WARNING] Skipping ECG model load — TensorFlow not available.")
        
    print("Models loaded successfully.")

def get_xgb_model():
    """Returns the loaded XGBoost model."""
    if _xgb_model is None:
        load_models()
    return _xgb_model

def get_ecg_model():
    """Returns the loaded ECG CNN model."""
    if _ecg_model is None:
        load_models()
    return _ecg_model
