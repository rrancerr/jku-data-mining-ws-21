import shelve

import pandas as pd
import streamlit as st
# vars
from sklearn.preprocessing import StandardScaler

WIND_DIRS = ('N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW')

# ui
st.title('Rain prediction')
st.subheader('Enter your data and predict whether it will rain tomorrow or not!')
# input fields
inp_date = st.date_input('Date')
inp_location = st.text_input('Location')
inp_min = st.text_input('MinTemp')
inp_max = st.text_input('MaxTemp')
inp_rainfall = st.text_input('Rainfall')
inp_evaporation = st.text_input('Evaporation')
inp_sunshine = st.text_input('Sunshine')
inp_windgustdir = st.selectbox('WindGustDir', (WIND_DIRS))
inp_windgustspeed = st.text_input('WindGustSpeed')
inp_winddir9am = st.selectbox('WindDir9am', WIND_DIRS)
inp_winddir3pm = st.selectbox('WindDir3pm', WIND_DIRS)
inp_windspeed9am = st.text_input('WindSpeed9am')
inp_windspeed3pm = st.text_input('WindSpeed3pm')
inp_humidity9am = st.text_input('Humidity9am')
inp_humidity3pm = st.text_input('Humidity3pm')
inp_pressure9am = st.text_input('Pressure9am')
inp_pressure3pm = st.text_input('Pressure3pm')
inp_cloud9am = st.text_input('Cloud9am')
inp_cloud3pm = st.text_input('Cloud3pm')
inp_temp9am = st.text_input('Temp9am')
inp_temp3pm = st.text_input('Temp3pm')
inp_raintoday_bool = st.checkbox('RainToday')

# get data of shelf
s = shelve.open('../shelf-data')
rf_mod = s['rf_mod']
X_train = s['X_train']
encoding_windGustDir = s['WindGustDir']
encoding_windDir9am = s['WindDir9am']
encoding_windDir3pm = s['WindDir3pm']
encoding_location = s['Location']

if st.button('Predict rain'):
    try:
        # get input data
        input_data = {'Date': [inp_date],
                      'Location': [inp_location],
                      'MinTemp': [inp_min],
                      'MaxTemp': [inp_max],
                      'Rainfall': [inp_rainfall],
                      'Evaporation': [inp_evaporation],
                      'Sunshine': [inp_sunshine],
                      'WindGustDir': [inp_windgustdir],
                      'WindGustSpeed': [inp_windgustspeed],
                      'WindDir9am': [inp_winddir9am],
                      'WindDir3pm': [inp_winddir3pm],
                      'WindSpeed9am': [inp_windspeed9am],
                      'WindSpeed3pm': [inp_windspeed3pm],
                      'Humidity9am': [inp_humidity9am],
                      'Humidity3pm': [inp_humidity3pm],
                      'Pressure9am': [inp_pressure9am],
                      'Pressure3pm': [inp_pressure3pm],
                      'Cloud9am': [inp_cloud9am],
                      'Cloud3pm': [inp_cloud3pm],
                      'Temp9am': [inp_temp9am],
                      'Temp3pm': [inp_temp3pm],
                      'RainToday': [inp_raintoday_bool]
                      }

        if not input_data['RainToday']:
            input_data['RainToday'] = 'No'
        else:
            input_data['RainToday'] = 'Yes'

        # create dataframe
        input_df = pd.DataFrame(data=input_data)

        # rearrange date
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        input_df['year'] = input_df['Date'].dt.year
        input_df['month'] = input_df['Date'].dt.month
        input_df['day'] = input_df['Date'].dt.day
        input_df.drop('Date', axis=1, inplace=True)

        # encode data
        input_df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
        input_df['WindGustDir'].replace(encoding_windGustDir, inplace=True)
        input_df['WindDir9am'].replace(encoding_windDir9am, inplace=True)
        input_df['WindDir3pm'].replace(encoding_windDir3pm, inplace=True)
        input_df['Location'].replace(encoding_location, inplace=True)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train[0:])
        X_test = scaler.transform(input_df)

        y_pred = rf_mod.predict(input_df.to_numpy())
        print(input_df.to_numpy())
        print(y_pred)
        temp_str = 'error'
        if y_pred[0] == 0:
            temp_str = 'not'
        elif y_pred[0] == 1:
            temp_str = ''

        if temp_str == 'error':
            raise Exception(f'y_pred invalid: {y_pred}')

        st.write(f'It\'ll {temp_str} rain tomorrow!')
    except Exception as error:
        st.write('Something went wrong, check your inputs!')
        raise error
