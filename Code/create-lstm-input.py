import pandas as pd
import datetime

## load data
taylor = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/taylor-combined.csv", index_col="date")
blue_mesa = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/blue_mesa-combined.csv", index_col="date")
fontenelle = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/fontenelle-combined.csv")
flaming_gorge = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/flaming_gorge-combined.csv")
navajo = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/navajo-combined.csv")
lake_powell = pd.read_csv("C:/Users/375237/Desktop/CRB-human-impacts/Data/lake_powell-combined.csv")

## remove rows before 1950 and after 2011
taylor.index = pd.to_datetime(taylor.index)
taylor = taylor.loc[datetime.date(year=1950,month=10,day=1):datetime.date(year=2011,month=12,day=31)]
blue_mesa.index = pd.to_datetime(blue_mesa.index)
blue_mesa = blue_mesa.loc[datetime.date(year=1950,month=10,day=1):datetime.date(year=2011,month=12,day=31)]

## percentage of missing values
print(taylor)
print('\n', blue_mesa)
## split into train and test data, convert to tensors, save results
