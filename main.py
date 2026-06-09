import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

from shap_analyzer import analyze_with_shap
from expert_system import generate_arabic_medical_report

def main():
    data = load_wine(as_frame=True)
    X, y = data.data, data.target
    class_names = list(data.target_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    sample_idx = 0

    patient_context = {
        "age": 67,
        "gender": "male",
        "comorbidities": ["diabetes"]
    }

    print("==================================================")
    print("1. Tree-Based Model (RandomForestClassifier)")
    print("==================================================")
    model_tree = RandomForestClassifier(n_estimators=100, random_state=42)
    model_tree.fit(X_train, y_train)
    
    shap_dict, sample_dict, pred = analyze_with_shap(
        model_tree, X_train, X_test.iloc[[sample_idx]]
    )
    report_tree = generate_arabic_medical_report(
        shap_dict, sample_dict, pred, 
        patient_context=patient_context, 
        class_names=class_names
    )
    print(report_tree)

    print("\n" + "==================================================")
    print("2. Linear Model (Logistic Regression)")
    print("==================================================")
    model_gen = LogisticRegression(max_iter=1000)
    model_gen.fit(X_train_scaled, y_train)
    
    shap_dict, sample_dict, pred = analyze_with_shap(
        model_gen, X_train_scaled, X_test_scaled.iloc[[sample_idx]]
    )
    report_linear = generate_arabic_medical_report(
        shap_dict, sample_dict, pred, 
        patient_context=patient_context, 
        class_names=class_names
    )
    print(report_linear)

    print("\n" + "==================================================")
    print("3. Deep Learning Model (Keras)")
    print("==================================================")
    model_dl = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model_dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model_dl.fit(X_train_scaled, y_train, epochs=10, verbose=0)
    
    raw_keras_predict = model_dl.predict
    
    def dl_predict_wrapper(x, **kwargs):
        preds = raw_keras_predict(x, verbose=0)
        return np.argmax(preds, axis=1)
    
    model_dl.predict = dl_predict_wrapper
    model_dl.predict_proba = raw_keras_predict
    
    shap_dict, sample_dict, pred = analyze_with_shap(
        model_dl, X_train_scaled, X_test_scaled.iloc[[sample_idx]]
    )
    report_dl = generate_arabic_medical_report(
        shap_dict, sample_dict, pred, 
        patient_context=patient_context, 
        class_names=class_names
    )
    print(report_dl)
    print("==================================================")

if __name__ == "__main__":
    main()