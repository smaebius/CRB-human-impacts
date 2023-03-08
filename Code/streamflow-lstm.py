import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import math
from scipy.stats import pearsonr
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import datetime

### Use GPU if possible
device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    print(torch.cuda.get_device_name())
    device = 'cuda'
else:
    print("No GPU available! Running on CPU")

### Load streamflow data
df = pd.read_csv('/content/drive/MyDrive/CRB-analysis/livneh/blue_mesa-combined-cluster.csv', index_col='date')
df.index = pd.to_datetime(df.index)

df = df.interpolate(method='spline', order=1, limit=10, limit_direction='both')

df = df.loc[datetime.date(year=1950,month=1,day=1):datetime.date(year=1962,month=1,day=1)]

# add previous day streamflow column
q_cms_lag = [df['q_cms'].shift(i) for i in range(1, 2)]
df['q_cms_lag'] = q_cms_lag[0]
df.fillna(0, inplace=True)

Xysplit = int(len(df)*0.9)

### Create LSTM class with two connected layers
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_size, 512)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.1)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) # internal state

        output, (hn, cn) = self.lstm(self.dropout(x), (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

### Create Metric functions
def compute_nse(observed, simulated):
  denominator = np.sum((observed - np.mean(observed)) ** 2)
  numerator = np.sum((simulated - observed) ** 2)
  nse_val = 1 - numerator / denominator
  return nse_val

def compute_rmse(observed, simulated):
  mse = np.square(np.subtract(simulated, observed)).mean()
  return math.sqrt(mse)

def compute_kge(observed, simulated):
  observed = observed.flatten()
  simulated = simulated.flatten()
  cc = pearsonr(observed, simulated)[0]
  rm = np.mean(observed)
  cm = np.mean(simulated)
  rd = np.std(observed)
  cd = np.std(simulated)
  root = np.square((cc-1)**2 + (cd/rd - 1)**2 + (cm/rm -1)**2)
  return 1 - root

def compute_log_nse(nse):
  if nse < 0:
    return -1000000
  else:
    return math.log(nse)

### Create Dataloader
class StreamflowTensorDataset(TensorDataset):
  def __init__(self, df, seq_length=1):
    self.df = df
    self.seq_length = seq_length
    self.X, self.y, self.mm, self.ss = self.data_to_tensor(self.df)

  def data_to_tensor(self, df):
    mm = MinMaxScaler()
    ss = StandardScaler()

    X_ss = ss.fit_transform(df.drop('q_cms', axis=1)) # independent var: forcings data 
    y_mm = mm.fit_transform(df[['q_cms']]) # dependent var: streamflow data

    X_tensors = Variable(torch.Tensor(X_ss))

    y = Variable(torch.Tensor(y_mm))

    X = torch.reshape(X_tensors,   (X_tensors.shape[0], 1, X_tensors.shape[1]))

    return X, y, mm, ss

  def __len__(self):
    return len(self.X)

  def __getitem__(self, i):
    if i >= self.seq_length - 1:
      i_start = i - self.seq_length + 1
      x = self.X[i_start:(i+1), :].squeeze(1)
    else:
      padding = self.X[0].repeat(self.seq_length-i-1, 1)
      x = self.X[0:(i+1), :]
      x = torch.cat((padding, x.squeeze(1)), 0)
    return x, self.y[i]

### Dataloader
dataset = StreamflowTensorDataset(df[0:Xysplit], seq_length=1)
dataloader = DataLoader(dataset, batch_size=len(df[0:Xysplit]), shuffle=False, num_workers=0)
X, y = next(iter(dataloader))

### Train LSTM
num_epochs = 5000
learning_rate = 5e-5

input_size = 13 # number of features
hidden_size = 7 # number of features in hidden state
num_layers = 1 # number of stacked lstm layers
num_classes = 1 # number of output classes 

lstm_model = LSTM1(num_classes, input_size, hidden_size, num_layers).to(device)

criterion = torch.nn.MSELoss() # regression
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  lstm_model.train()
  train_loss, train_accuracy = 0, 0

  for i, (x, y) in enumerate(dataloader):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # gradient calculation
    outputs = lstm_model.forward(x) # forward pass

    # loss function
    loss = criterion(outputs, y)
    train_loss += loss
    loss.backward()

    # torch.nn.utils.clip_grad_norm(parameters=lstm_model.parameters(), max_norm=10, norm_type=2.0)
    optimizer.step() # backpropagation
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

### Make predictions
df_X_ss = dataset.ss.transform(df.drop(['q_cms'], axis=1))
df_y_mm = dataset.mm.transform(df[['q_cms']])

df_X_ss = Variable(torch.Tensor(df_X_ss)).to(device)
df_y_mm = Variable(torch.Tensor(df_y_mm)).to(device)

df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

train_predict = lstm_model(df_X_ss) # forward pass
y_predicted = train_predict.cpu().data.numpy()
df_y = df_y_mm.cpu().data.numpy()

y_predicted = dataset.mm.inverse_transform(y_predicted) # reverse transformation
df_y = dataset.mm.inverse_transform(df_y)

df.index = pd.to_datetime(df.index)
dates = df.index.to_numpy()

test_nse = compute_nse(df_y[Xysplit:], y_predicted[Xysplit:])
test_rmse = compute_rmse(df_y[Xysplit:], y_predicted[Xysplit:])
test_kge = compute_kge(df_y[Xysplit:], y_predicted[Xysplit:])

print('NSE: ', test_nse, ' Log NSE: ', compute_log_nse(test_nse))
print('RMSE: ', test_rmse)
print('KGE: ', test_kge)

### Plot results
plt.figure(figsize=(20,8))
plt.axvline(x=np.datetime64('1962-01-01T00:00:00.000000000'), c='g', linestyle='-.', label='dam construction start')
plt.axvline(x=dates[Xysplit], c='r', linestyle='--', label='train test split') # train test split
plt.plot(dates, df_y, label='Observed Streamflow (cms)')
plt.plot(dates, y_predicted, label='Predicted USGS (cms)')
plt.title('LSTM Prediction for Pre-Dam Blue Mesa Reservoir')
plt.legend()
plt.show()