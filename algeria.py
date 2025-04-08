import numpy as np
import streamlit as st
import pandas as pd

import seaborn as sns

import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

fire = pd.read_csv('Algerian_forest_fires_dataset.csv')
# fire[fire.isnull().any(axis=1)]
fire.loc[:121, "Region"] = 0
fire.loc[125:, "Region"] = 1
fire.drop([122,123,124], inplace= True)
# fire.reset_index(drop=True)
# fire.isnull().any(axis=1)
# fire[fire.isnull().any(axis=1)]
fire.drop([168], inplace= True)
# fire.columns
fire.columns = fire.columns.str.strip() # to remove white space
# fire['Classes']
fire['Classes'] = fire['Classes'].str.strip()
# fire['Classes'].unique
fire['Classes'] = fire['Classes'].map({'not fire': 1, 'fire': 0})
# fire.columns
fire[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC',
       'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region']] = fire[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC',
       'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes', 'Region']].astype(float).round().astype(int)

x = fire.drop(['day', 'month', 'year', 'FWI'], axis=1)
y = fire['FWI']

train_test_split(x,y,test_size=0.3,random_state=42)
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)


st.title("Algerian Forest Fires Prediction")

lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
prediction = lr_model.predict(x_test)

from joblib import dump, load
dump(lr_model, 'forest_fires_model.joblib') 
load_model = load('forest_fires_model.joblib')

Temperature = st.sidebar.number_input("Temperature", min_value=0.0, value=0.0)
RH 	=  st.sidebar.number_input("RH", min_value=0.0, value=0.0)
Ws 	= st.sidebar.number_input("WS", min_value=0.0, value=0.0)
Rain = st.sidebar.number_input("Rain", min_value=0.0, value=0.0)	
FFMC = st.sidebar.number_input("FFMC",min_value=0.0, value=0.0)	
DMC  = st.sidebar.number_input("DMC", min_value=0.0, value=0.0)
DC = st.sidebar.number_input("DC", min_value=0.0, value=0.0)
ISI = st.sidebar.number_input("ISI", min_value=0.0, value=0.0)	
BUI = st.sidebar.number_input("BUI")	
Classes = st.sidebar.number_input("Classes",min_value=0.0, value=0.0)	
Region = st.sidebar.number_input("Region", min_value=0.0, value=0.0 )

input_data = np.array ([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Classes, Region]])

if st.button("Predict Fires"):
    prediction = lr_model.predict(input_data)
    st.success(f"Predicted Fires: {prediction[0]:.2f}")

# input_data = np.array(input_data)
# input = input_data.reshape(1,-1)

# st.write('Prediction')
# if st.button('Predict'):
#     prediction = lr_model.predict(input)
#     st.write(prediction)

