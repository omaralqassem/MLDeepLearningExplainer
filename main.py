import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Input

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Input

from shap_analyzer import analyze_with_shap

from expert_system import (
    generate_arabic_medical_report,
    classify_features_clinical,
    displaychart,
    display_interactiveChart
)

def main():
    data = pd.read_csv('diabetes.csv')
    
    # Pregnancies: Number of times pregnant
    # Glucose: Plasma glucose concentration two hours after an oral glucose tolerance test
    # BloodPressure: Diastolic blood pressure (mm Hg)
    # SkinThickness: Triceps skinfold thickness (mm)
    # Insulin: 2-Hour serum insulin (mu U/ml)
    # BMI: Body mass index (weight in kg/(height in m)^2)
    # DiabetesPedigreeFunction: Diabetes pedigree function
    # Age: Age in years
    # Outcome: Class variable (0 for absence, 1 for presence of diabetes)
            
    dataset_new = data
    dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 
    
    # Replacing NaN with mean values
    dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
    dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
    dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
    dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
    dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)
            
    X = dataset_new.drop(columns=['Outcome'])
    y = dataset_new['Outcome']
    class_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # scaler = StandardScaler()
    # X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    # X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    sample_idx = 23
    print(X_test.iloc[sample_idx])

    patient_context = {
        "age": int(data["Age"].iloc[sample_idx]),
        "gender": "female",
        "comorbidities": ["diabetes"]
    }

    # print("==================================================")
    # print("1. Tree-Based Model (RandomForestClassifier)")
    # print("==================================================")
    # model_tree = RandomForestClassifier(n_estimators=100, random_state=42)
    # model_tree.fit(X_train, y_train)
    
    # shap_dict, sample_dict, pred = analyze_with_shap(
    #     model_tree, X_train, X_test.iloc[[sample_idx]]
    # )
    # report_tree = generate_arabic_medical_report(
    #     shap_dict, sample_dict, pred, 
    #     patient_context=patient_context, 
    #     class_names=class_names
    # )
    # print(report_tree)
    # print("\n[Generating Charts for RandomForestClassifier...]")
    # classified_tree = classify_features_clinical(shap_dict, sample_dict, patient_context)
    # displaychart(classified_tree, "RandomForest Classifier")
    # display_interactiveChart(classified_tree, "RandomForest Classifier")

    print("\n" + "==================================================")
    print("2. Linear Model (Logistic Regression)")
    print("==================================================")
    model_gen = LogisticRegression()
    model_gen.fit(X_train, y_train)
    
    shap_dict, sample_dict, pred = analyze_with_shap(
        model_gen, X_train, X_test.iloc[[sample_idx]]
    )
    report_linear = generate_arabic_medical_report(
        shap_dict, sample_dict, pred, 
        patient_context=patient_context, 
        class_names=class_names
    )
    print(report_linear)
    print("\n[Generating Charts for Logistic Regression...]")
    classified_linear = classify_features_clinical(shap_dict, sample_dict, patient_context)
    displaychart(classified_linear, "Logistic Regression")
    display_interactiveChart(classified_linear, "Logistic Regression")

    # print("\n" + "==================================================")
    # print("3. Deep Learning Model (Keras)")
    # print("==================================================")
    # model_dl = Sequential([
    #     Input(shape=(X_train.shape[1],)),
    #     Dense(16, activation='relu'),
    #     Dense(3, activation='softmax')
    # ])
    # model_dl.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # model_dl.fit(X_train_scaled, y_train, epochs=10, verbose=0)
    
    # raw_keras_predict = model_dl.predict
    
    # def dl_predict_wrapper(x, **kwargs):
    #     preds = raw_keras_predict(x, verbose=0)
    #     return np.argmax(preds, axis=1)
    
    # model_dl.predict = dl_predict_wrapper
    # model_dl.predict_proba = raw_keras_predict
    
    # shap_dict, sample_dict, pred = analyze_with_shap(
    #     model_dl, X_train_scaled, X_test_scaled.iloc[[sample_idx]]
    # )
    # report_dl = generate_arabic_medical_report(
    #     shap_dict, sample_dict, pred, 
    #     patient_context=patient_context, 
    #     class_names=class_names
    # )
    # print(report_dl)
    
    # print("\n[Generating Charts for Keras Deep Learning Model...]")
    # classified_dl = classify_features_clinical(shap_dict, sample_dict, patient_context)
    # displaychart(classified_dl, "Keras Model")
    # display_interactiveChart(classified_dl, "Keras Model")
    # print("==================================================")

if __name__ == "__main__":
    main()