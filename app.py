import pyrebase
import requests 
import pandas as pd
import streamlit as st
from datetime import datetime
import hydralit as hy
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import datetime
from keras.models import load_model
from streamlit_lottie import st_lottie
import requests 


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
  
# Configuration Key 
firebaseConfig = {
  'apiKey': "AIzaSyDy5N-2aJ9B-q_XzkmMh2Y_i2wYtCZ-nmI",
  'authDomain': "test-firestore-streamlit-5e52b.firebaseapp.com",
  'projectId': "test-firestore-streamlit-5e52b",
  'databaseURL' : "https://test-firestore-streamlit-5e52b-default-rtdb.firebaseio.com/",
  'storageBucket': "test-firestore-streamlit-5e52b.appspot.com",
  'messagingSenderId': "337653340318",
  'appId': "1:337653340318:web:7e79c9670fb1453314038f",
  'measurementId': "G-N7EYV8VVBW"
}

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

#Database
db = firebase.database()
storage = firebase.storage()
st.sidebar.title("Stock Price Prediction App")


#Authentication
choice = st.sidebar.selectbox('Login/Singup', ['Login', 'Singup'])

email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type='password')

if choice == 'Singup':
  handle = st.sidebar.text_input('Please input ypur app handle name', value='Default')
  submit = st.sidebar.button('Create my Account')
  
  if submit:
    user = auth.create_user_with_email_and_password(email, password)
    st.success('Your account is created successfully!')
    st.balloons()    
    #Sign in
    user = auth.sign_in_with_email_and_password(email, password)
    db.child(user['localId']).child("Handle").set(handle)
    db.child(user['localId']).child("ID").set(user['localId'])
    st.title('Welcome' + handle)
    st.info('Login via login drop down selection')
    
if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        app = hy.HydraApp(title='Stock Price Prediction',favicon="🐙",hide_streamlit_markers=True,use_navbar=True, navbar_sticky=True)


        def load_lottieurl(url: str):
            r = requests.get(url)
            if r.status_code != 200:
                return None
            return r.json()

        @app.addapp(is_home=True)
        def my_home():
            st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<link rel="stylesheet" href="./style/style.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
                st.markdown("""
                    <div class="container" style="padding-top:5%">
                        <div class="row">
                            <div class="col">
                                <p style="font-size:40px; font-weight:600; margin-top:100px">Stock Market</p>
                                <p>Confused about how much to invest ? Difficult to understand stocks ? Here You go...! Find somethings really interesting about bear nd bull of the market from the pervious data. This app will discover the future value of company stock and other financial assets traded on an exchange</p>
                            </div>
                        </div>
                    </div>
                    <br>                   
                """,unsafe_allow_html=True)      
            with col2:    
                lottie_bot = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kuhijlvx.json")
                
                st_lottie(
                lottie_bot,
                speed=1,
                reverse=False,
                loop=True,
                quality="low", # medium ; high
                height="10dp",
                width="5dp",
                key=None,
                )

        @app.addapp()
        def Stocks():
            option = st.sidebar.selectbox('Select one symbol', ( 'AAPL', 'MSFT',"SPY",'WMT'))

            today = datetime.date.today()
            before = today - datetime.timedelta(days=700)
            start_date = st.sidebar.date_input('Start date', before)
            
            end = today - datetime.timedelta(days=200) 
            end_date = st.sidebar.date_input('End date', end)
            
            future = end_date + datetime.timedelta(days=90)
            future_date = st.sidebar.date_input('Future date', future)
            
            if start_date < end_date:
                st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
            else:
                st.sidebar.error('Error: End date must fall after start date.')

           
            new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Stock Price Prediction</p>'
            st.markdown(new_title, unsafe_allow_html=True)


            #st.title('Stock Price Prediction')

            user_input = st.text_input('Enter Stock Ticker', option)
            df = data.DataReader(user_input, 'yahoo', start_date, end_date)
            df2 = data.DataReader(user_input, 'yahoo', end_date, future_date)

            #Describing Data
            st.subheader('Data')
            st.write(df.describe())

            #Visualizations
            st.subheader('Closing Price vs Time Chart')
            fig = plt.figure(figsize = (12,6))
            plt.plot(df.Close)
            st.pyplot(fig)

            st.subheader('Closing Price vs Time Chart with 100MA')
            ma100 = df.Close.rolling(100).mean()
            fig = plt.figure(figsize = (12,6))
            plt.plot(ma100)
            plt.plot(df.Close)
            st.pyplot(fig)

            st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
            ma100 = df.Close.rolling(100).mean()
            ma200 = df.Close.rolling(200).mean()
            fig = plt.figure(figsize = (12,6))
            plt.plot(ma100)
            plt.plot(ma200)
            plt.plot(df.Close)
            st.pyplot(fig)
            
            #Spliting data into Training and Testing
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #starting from 0th index to 70% of the total values
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])
            data_future = pd.DataFrame(df2['Close'])
            
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0,1))

            data_training_array = scaler.fit_transform(data_training)
            future_array = scaler.fit_transform(data_future)
            future_array = np.array(future_array)
            
            #Load my Model
            model = load_model('keras_model.h5')

            #Testing part
            past_100_days = data_training.tail(100)
            final_df = past_100_days.append(data_testing, ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100: i])
                y_test.append(input_data[i, 0])
                
            x_test, y_test = np.array(x_test), np.array(y_test)
            y_predicted = model.predict(x_test)
            scaler = scaler.scale_

            scale_factor = 1/scaler[0]
            y_predicted = y_predicted * scale_factor
            y_test = y_test * scale_factor
            
            #Future Predicition
            pred_future = model.predict(future_array)
            pred_future = pred_future * scale_factor


            #Final Graph
            st.subheader('Predictions vs Original')
            fig2 = plt.figure(figsize=(12,6))
            plt.plot(y_test, 'b', label = 'Original Price')
            plt.plot(y_predicted, 'r', label = 'Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend(loc='best')
            st.pyplot(fig2)
            
            #Final Graph for Future Predictions
            st.subheader('Future Predictions')
            fig3 = plt.figure(figsize=(12,6))
            plt.plot(pred_future, 'g', label = 'Future Predicted Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend(loc='best')
            st.pyplot(fig3)
            
        app.run()