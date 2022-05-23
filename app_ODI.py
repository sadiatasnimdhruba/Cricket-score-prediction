import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor
from scipy import sparse
from sklearn import svm
from sklearn import datasets
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorIndexer, OneHotEncoder, VectorAssembler

pipe_ODI = pickle.load(open('pipe_ODI.pkl', 'rb'))

teams = ['Australia',
         'Zimbabwe',
         'India',
         'Bangladesh',
         'New Zealand',
         'South Africa',
         'England',
         'West Indies',
         'Afghanistan',
         'Pakistan',
         'Sri Lanka']

cities = ['Mirpur',
          'London',
          'Colombo',
          'Harare',
          'Bulawayo',
          'Sydney',
          'Abu Dhabi',
          'Rangiri',
          'Melbourne',
          'Centurion',
          'Sharjah',
          'Adelaide',
          'Perth',
          'Birmingham',
          'Dubai',
          'Auckland',
          'Johannesburg',
          'Brisbane',
          'Pallekele',
          'Wellington',
          'Cardiff',
          'Durban',
          'Hamilton',
          'Port Elizabeth',
          'Manchester',
          'Nottingham',
          'Southampton',
          'Antigua',
          'Cape Town',
          'Guyana',
          'Trinidad',
          'Karachi',
          'Christchurch',
          'Hambantota',
          'Leeds',
          'Chandigarh',
          'Napier',
          'Jamaica',
          'St Lucia',
          'Lahore',
          'Mumbai',
          'St Kitts',
          'Chester-le-Street',
          'Chittagong',
          'Ahmedabad',
          'Barbados',
          'Dhaka',
          'Grenada',
          'Hobart',
          'Jaipur',
          'Delhi',
          'Mount Maunganui',
          'Nagpur',
          'Visakhapatnam',
          'Chennai',
          'Bloemfontein',
          'Dunedin',
          'Fatullah',
          'St Vincent',
          'Bristol',
          'Nelson',
          'Canberra',
          'Rajkot',
          'Kolkata',
          'Hyderabad',
          'Kanpur',
          'Potchefstroom',
          'Cuttack',
          'Kuala Lumpur',
          'Indore',
          'Pune',
          'Rawalpindi',
          'Paarl',
          'Multan']

st.title('One Day Cricket Score Predictor')

col1, col2 = st.beta_columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.beta_columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>10)')
with col5:
    wickets = st.number_input('Wickets out')

last_ten = st.number_input('Runs scored in last 10 overs')

if st.button('Predict Score'):
    balls_left = 600 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs

    input_df = pd.DataFrame(
        {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': city, 'current_score': [current_score],
         'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_ten': [last_ten]})
    st.table(input_df)
    result = pipe_ODI.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))


