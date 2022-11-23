import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from pycaret.classification import *
import datetime
import pickle
import xgboost as xgb

st.subheader('Churn Model Prediction')

st.sidebar.header('User input Parameters')

def user_input_feature():
    CreditScore = st.sidebar.slider('CreditScore', 350, 850, 500)
    Geography=st.sidebar.selectbox('Geography',['France','Spain','Germany'])
    Gender=st.sidebar.selectbox('Gender',['Male','Female'])
    Age = st.sidebar.slider('Age', 18, 100, 30)
    Tenure = st.sidebar.slider('Tenure', 0, 10, 5)
    Balance = st.sidebar.text_input('Balance','10000')    
    NumOfProducts = st.sidebar.slider('Number Of Products', 1, 4, 1)
    HasCrCard=st.sidebar.selectbox('Has Credit Card ?',['Yes', 'No'])
    IsActiveMember=st.sidebar.selectbox('Is Active Member ?',['Yes', 'No'])
    EstimatedSalary = st.sidebar.text_input('Estimated Salary','1000')  

    Algorithm=st.sidebar.selectbox('Select Algorithm',['XGBoosClassifier','Random Forest Classifier','Decision Tree Classifier','SVM - Linear Kernel','Logistic Regression'])

    mapping_Yes_No={'No': 0, 'Yes': 1}
    mapping_Gender={'Male': 0, 'Female': 1}
    mapping_Geography={'France' : 0,'Spain' : 1, 'Germany' : 2}
    dt={'CreditScore' : CreditScore,
        'Geography': mapping_Geography[Geography],
        'Gender': mapping_Gender[Gender],
        'Age' :  Age,
        'Tenure' : Tenure,
        'Balance' :  float(Balance),
        'NumOfProducts' : NumOfProducts,
        'HasCrCard': mapping_Yes_No[HasCrCard],
        'IsActiveMember': mapping_Yes_No[IsActiveMember],
        'EstimatedSalary' :  float(EstimatedSalary),
       }

    lb={'CreditScore' : CreditScore,
        'Geography': Geography,
        'Gender': Gender,
        'Age' :  Age,
        'Tenure' : Tenure,
        'Balance' :  Balance,
        'NumOfProducts' : NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary' :  EstimatedSalary,
       }

    features=pd.DataFrame(dt,index=[0])
    labels=pd.DataFrame(lb,index=[0])

    return features,labels,Algorithm

df,lb,Algo=user_input_feature()
mapping_1_0={'0': 'No', '1': 'Yes'}

st.subheader('User input Parameters :')
st.write(lb)

if Algo=='XGBoosClassifier':
    loaded_Model=pickle.load(open(Algo+'.pkl', 'rb'))
    prediction = loaded_Model.predict(df)
    st.subheader(Algo +' prediction')
    st.subheader('Exit :  ' +  mapping_1_0[str(prediction.item())])

else: 
    loaded_Model=load_model(Algo)
    prediction = predict_model(loaded_Model, data = df)
    st.subheader(Algo +' prediction')
    st.subheader('Exit :  ' +  mapping_1_0[str(prediction['Label'].item())])



