import numpy as np
from numpy import newaxis
import pandas as pd
import matplotlib.pyplot as plt


class Predict_Future:

    def make_window(self, sequence_length, train_test_split, return_original_x = True):        
        # Create the initial results df with a look_back of 60 days
        result = []
        
        # 3D Array
        for index in range(len(self.rnn_df) - sequence_length):
            result.append(self.rnn_df[self.numeric_colname][index: index + sequence_length])  
        
        # Getting the initial train_test split for our min/max val scalar
        train_test_split = 0.9
        row = int(round(train_test_split * np.array(result).shape[0]))
        train = np.array(result)[:row, :]
        x_train = train[:, :-1]
        
        # Manual MinMax Scaler
        X_min = x_train.min()
        X_max = x_train.max()
        
        # keep the originals in case
        X_min_orig = x_train.min()
        X_max_orig = x_train.max()
        
        # Minmax scaler and a reverse method
        def minmax(X):
            return (X-X_min) / (X_max - X_min)
        
        def reverse_minmax(X):
            return X * (X_max-X_min) + X_min
        
        def minmax_windows(window_data):
            normalised_data = []
            for window in window_data:
                window.index = range(sequence_length)
                normalised_window = [((minmax(p))) for p in window]
                normalised_data.append(normalised_window)
            return normalised_data
    
        result = minmax_windows(result)
        # Convert to 2D array
        result = np.array(result)
        if return_original_x:
            return result, X_min_orig, X_max_orig
        else:
            return result
        
    def __init__(self, x_test, lstm_model):
        self.x_test = x_test
        self.lstm_model = lstm_model
        
        
    def predict_future(self, X_min, X_max, numeric_colname, timesteps_to_predict, return_future = True):
    
        curr_frame = self.x_test[len(self.x_test)-1]
        future = []
        
        for i in range(timesteps_to_predict):
              # append the prediction to our empty future list
             future.append(self.lstm_model.predict(curr_frame[newaxis,:,:])[0,0])
              # insert our predicted point to our current frame
             curr_frame = np.insert(curr_frame, len(self.x_test[0]), future[-1], axis=0)
              # push the frame up one to make it progress into the future
             curr_frame = curr_frame[1:]
        
        def reverse_minmax(X, X_max = X_max, X_min = X_min):
            return X * (X_max-X_min) + X_min

        # Reverse the original frame and the future frame
        reverse_curr_frame = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in self.x_test[len(self.x_test)-1]],
                                           "historical_flag":1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                           "historical_flag":0})
        
        # Change the indicies to show prediction next to the actuals in orange
        reverse_future.index += len(reverse_curr_frame)
        
        print("See Plot for Future Predictions")
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Future of "+ str(timesteps_to_predict) + " days")
        plt.show()
        
        if return_future:
            return reverse_future

    
    
    
