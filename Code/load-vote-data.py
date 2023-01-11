### Download VotE USGS Data and convert to cubic meters per second
from VotE.streamflow import export_streamflow as es
from VotE import config; config.vote_db()
import VotE.sql_ops as sops
from VotE.streamflow import pg_sf_table_metadata as sfmd
import pandas as pd
import os

condition = "WHERE id_source in ('09109000', '09128000', '09211200', '09234500', '09355500', '09380000')"
query_cols = ['id_gage', 'id_source', 'river_name_source']
df = sops.query_table(condition, sfmd.gages(), query_cols=query_cols)
print(df)

# Fetch streamflow data
sf = es.get_streamflow_timeseries(df['id_gage'].values[:], expand=True)
print(sf.columns)

taylor_park = sf[sf['id_gage']==15324061417]
blue_mesa = sf[sf['id_gage']==15106909764]
fontenelle = sf[sf['id_gage']==14605611574]
flaming_gorge = sf[sf['id_gage']==14736634909]
navajo = sf[sf['id_gage']==15096645837]
lake_powell = sf[sf['id_gage']==-14284587762]

dir_path = 'C:/Users/375237/Desktop/CRB-human-impacts/Data/'
def clean_and_save_vote_data(df, outpath):
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.q_cms
    df.to_csv(dir_path + outpath + '.csv')

clean_and_save_vote_data(taylor_park, "taylor_park")
clean_and_save_vote_data(blue_mesa, "blue_mesa")
clean_and_save_vote_data(fontenelle, "fontenelle")
clean_and_save_vote_data(flaming_gorge, "flaming_gorge")
clean_and_save_vote_data(navajo, "navajo")
clean_and_save_vote_data(lake_powell, "lake_powell")