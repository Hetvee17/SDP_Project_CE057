import hydralit as hy
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import pandas as pd
import streamlit as st
import datetime
from keras.models import load_model
import streamlit as st
from streamlit_lottie import st_lottie
import requests 
import LSTM_Prep

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
                        <p style="font-size:40px; font-weight:600; margin-top:100px">It’s a market of stocks</p>
                        <p>Confused about how much to invest ? Difficult to understand stocks ? Here You go...! Find somethings really interesting about bear nd bull of the market from the pervious data. This app will discover the future value of company stock and other financial assets traded on an exchange</p>
                    </div>
                </div>
            </div>
            <br>
            <div class="container">
            <div class="row">
            <div class="col">
            <h4>Stock....! Stock....! Stock....!</h4>
            <p> The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy.  </p>
            </div></div>
            </div>
        """,unsafe_allow_html=True)      
    with col2:    
        lottie_bot = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_ofa3xwo7.json")
        
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
def About():
    st.title('About')

    st.write('This is the `about page` of this multi-page app.')

    st.write('In this app, we will be building a simple classification model using the Iris dataset.')
 
@app.addapp()
def Stocks():
    option = st.sidebar.selectbox('Select one symbol', ( 'AAPL', 'MSFT',"SPY",'WMT'))

    today = datetime.date.today()
    before = today - datetime.timedelta(days=700)
    start_date = st.sidebar.date_input('Start date', before)
    end_date = st.sidebar.date_input('End date', today)
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')

    #hide_menu_style = """
    #                   <style>
    #                   #MainMenu {visibility : hidden; }
    #                  #footer {visibilitty : hidden; }
    #                 </style>
    #            """
    #st.markdown(hide_menu_style, unsafe_allow_html=True)

    new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Stock Price Prediction</p>'
    st.markdown(new_title, unsafe_allow_html=True)


    #st.title('Stock Price Prediction')

    user_input = st.text_input('Enter Stock Ticker', option)
    df = data.DataReader(user_input, 'yahoo', start_date, end_date)

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

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)


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

    #Final Graph
    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='best')
    st.pyplot(fig2)
    
    split = 0.8
    sequence_length = 60
    
    
    data_prep = LSTM_Prep.pd.DataFrame(dataset = df)
    rnn_df = data_prep.preprocess_rnn(date_colname = 'date', numeric_colname = 'perc', pred_set_timesteps = 60)


    series_prep = LSTM_Prep.Series_Prep(rnn_df =  rnn_df, numeric_colname = 'perc')
    window, X_min, X_max = series_prep.make_window(sequence_length = sequence_length, 
                                                train_test_split = split, 
                                                return_original_x = True)

    x_train, x_test, y_train, y_test = series_prep.reshape_window(window, train_test_split = split)

    future = LSTM_Prep.Predict_Future(x_test  = x_test, lstm_model = model)
    # Checking its accuracy on our training set
    #future.predicted_vs_actual(X_min = X_min, X_max = X_max, numeric_colname = 'perc')
    # Predicting 'x' timesteps out
    future.predict_future(X_min = X_min, X_max = X_max, numeric_colname = 'perc', timesteps_to_predict = 15, return_future = True)


    #Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()