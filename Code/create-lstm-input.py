import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import torch
from torch.autograd import Variable

## function to load and process data
def load_process_data(file_path):
    # load data
    df_name = file_path.split('.')[0].split('-')[-2].split('/')[-1]
    df = pd.read_csv(file_path, index_col="date")
    df.index = pd.to_datetime(df.index)

    # remove rows before 1950 and after 2011
    df = df.loc[datetime.date(year=1950,month=10,day=1):datetime.date(year=2011,month=12,day=31)]

    # percentage of missing values
    print(df_name, '\n', df.isnull().sum().sort_values(ascending=False)/len(df), '\n')
    print(df)

    # remove rows with na
    df.dropna(inplace=True)

    # add cumulative time delta column
    time_delta = df.index - df.index[0]
    df['time_delta'] = time_delta.days

    # train test split at 10%
    Xysplit = int(len(df)*0.9)
    
    # split into x and y inputs
    X = df.drop(columns=['q_cms'])
    y = df[['q_cms']]

    # scale data and convert to tensors
    mm = MinMaxScaler()
    ss = StandardScaler()

    X_ss = ss.fit_transform(X)
    y_mm = mm.fit_transform(y)

    X_train = X_ss[:Xysplit, :]
    X_test = X_ss[Xysplit:, :]

    y_train = y_mm[:Xysplit, :]
    y_test = y_mm[Xysplit:, :]

    # convert to tensors and reshape to LSTM format
    X_train_tensors = Variable(torch.Tensor(X_train))
    X_test_tensors = Variable(torch.Tensor(X_test))

    y_train_tensors = Variable(torch.Tensor(y_train))
    y_test_tensors = Variable(torch.Tensor(y_test))

    X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

    print("Training shape", X_train_tensors_final.shape, y_train_tensors.shape)
    print("Testing shape", X_test_tensors_final.shape, y_test_tensors.shape)

    # torch.save(X_train_tensors_final, 'C:/Users/375237/Desktop/CRB-human-impacts/Data/tensors/' + df_name + '_X_train.pt')
    # torch.save(y_train_tensors, 'C:/Users/375237/Desktop/CRB-human-impacts/Data/tensors/' + df_name + '_y_train.pt')
    # torch.save(X_test_tensors_final, 'C:/Users/375237/Desktop/CRB-human-impacts/Data/tensors/' + df_name + '_X_test.pt')
    # torch.save(y_test_tensors, 'C:/Users/375237/Desktop/CRB-human-impacts/Data/tensors/' + df_name + '_y_test.pt')


load_process_data("C:/Users/375237/Desktop/CRB-human-impacts/Data/taylor-combined.csv")
load_process_data("C:/Users/375237/Desktop/CRB-human-impacts/Data/blue_mesa-combined.csv")
load_process_data("C:/Users/375237/Desktop/CRB-human-impacts/Data/fontenelle-combined.csv")
load_process_data("C:/Users/375237/Desktop/CRB-human-impacts/Data/flaming_gorge-combined.csv")
load_process_data("C:/Users/375237/Desktop/CRB-human-impacts/Data/navajo-combined.csv")
load_process_data("C:/Users/375237/Desktop/CRB-human-impacts/Data/lake_powell-combined.csv")
