## training LSTM for Blue Mesa Reservoir
# Based on the tutorial: https://cnvrg.io/pytorch-lstm/

import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False

    return True

# create lstm class
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

def compute_nse(observed, simulated):
  denominator = np.sum((observed - np.mean(observed)) ** 2)
  numerator = np.sum((simulated - observed) ** 2)
  nse_val = 1 - numerator / denominator
  return nse_val

def lstm_train_and_predict(tensor_name, lstm_model, observed_df):
  X_train_tensors_final = torch.load(f'/content/drive/MyDrive/CRB-analysis/tensors/{tensor_name}_X_train.pt')
  y_train_tensors = torch.load(f'/content/drive/MyDrive/CRB-analysis/tensors/{tensor_name}_y_train.pt')
  X_test_tensors_final = torch.load(f'/content/drive/MyDrive/CRB-analysis/tensors/{tensor_name}_X_test.pt')
  y_test_tensors = torch.load(f'/content/drive/MyDrive/CRB-analysis/tensors/{tensor_name}_y_test.pt')

  seed = 42
  num_epochs = 1000
  learning_rate = 0.001

  criterion = torch.nn.MSELoss() # regression
  optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

  set_seed(seed)

  for epoch in range(num_epochs):
    outputs = lstm_model.forward(X_train_tensors_final) # forward pass
    optimizer.zero_grad() # gradient calculation
  
    # loss function
    loss = criterion(outputs, y_train_tensors)
  
    loss.backward()
  
    optimizer.step() # backpropagation
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

  # load data for predictions 
  mm = MinMaxScaler()
  ss = StandardScaler()

  df_X_ss = ss.fit_transform(observed_df.iloc[:, 1:]) # independent var: forcings data 
  df_y_mm = mm.fit_transform(observed_df.iloc[:, 0:1]) # dependent var: streamflow data

  df_X_ss = Variable(torch.Tensor(df_X_ss))
  df_y_mm = Variable(torch.Tensor(df_y_mm))

  df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

  train_predict = lstm_model(df_X_ss) # forward pass
  y_predicted = train_predict.data.numpy()
  df_y = df_y_mm.data.numpy()

  y_predicted = mm.inverse_transform(y_predicted) # reverse transformation
  df_y = mm.inverse_transform(df_y)

  observed_df.index = pd.to_datetime(observed_df.index)
  dates = observed_df.index.to_numpy()

  test_nse = compute_nse(df_y[15000:], y_predicted[15000:])
  all_nse = compute_nse(df_y, y_predicted)
  print('\nAll data NSE: ', all_nse)
  print('Test NSE: ', test_nse)

  return dates, df_y, y_predicted, test_nse

## train blue mesa lstm
X_train_tensors_final = torch.load('/content/drive/MyDrive/CRB-analysis/tensors/blue_X_train.pt')
tensor_name = 'blue'
input_size = X_train_tensors_final.shape[2] # forcings
hidden_size = 2
num_layers = 1 # stacked lstm layers
num_classes = 1
lstm_model = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
observed_df = pd.read_csv('/content/drive/MyDrive/CRB-analysis/blue_mesa.csv', index_col='datetime')

dates, df_y, y_predicted, test_nse = lstm_train_and_predict(tensor_name, lstm_model, observed_df)

plt.figure(figsize=(20,8))
plt.axvline(x=np.datetime64('1962-01-01T00:00:00.000000000'), c='g', linestyle='-.', label='dam construction start')
plt.axvline(x=np.datetime64('1966-01-01T00:00:00.000000000'), c='g', linestyle='-.', label='dam construction end')
plt.axvline(x=dates[15000], c='r', linestyle='--', label='train test split') # train test split
plt.plot(dates, df_y, label='Observed Streamflow (cms)')
plt.plot(dates, y_predicted, label='Predicted USGS (cms)')
plt.title('Vanilla LSTM Prediction for Blue Mesa Reservoir')
plt.legend()
plt.show()