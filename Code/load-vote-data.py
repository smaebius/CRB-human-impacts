### Download VotE USGS Data and convert to cubic meters per second
from VotE.streamflow import export_streamflow as es
from VotE import config; config.vote_db()
import VotE.sql_ops as sops
from VotE.streamflow import pg_sf_table_metadata as sfmd
import pandas as pd
import os

# condition = "WHERE id_source in ('09109000', '09128000', '09211200', '09234500', '09355500', '09380000')"
condition = "WHERE id_source in ('09109000', '09107000', '09107500', '09108000', '09128000', '09127000', '09126000', '09124500', '09211200', '09209500', '09209400', '09211000',  '09234500', '09229500', '09217000', '09224700', '09355500', '09355000', '09346400', '09342500', '09380000', '09379500', '09182400', '09379910')"
query_cols = ['id_gage', 'id_source', 'river_name_source']
df = sops.query_table(condition, sfmd.gages(), query_cols=query_cols)
print(df)

# Fetch streamflow data
sf = es.get_streamflow_timeseries(df['id_gage'].values[:], expand=True)
print(sf.columns)

# taylor_park = sf[sf['id_gage']==15324061417]
# blue_mesa = sf[sf['id_gage']==15106909764]
# fontenelle = sf[sf['id_gage']==14605611574]
# flaming_gorge = sf[sf['id_gage']==14736634909]
# navajo = sf[sf['id_gage']==15096645837]
# lake_powell = sf[sf['id_gage']==-14284587762]

# dir_path = 'C:/Users/375237/Desktop/CRB-human-impacts/Data/'
# def clean_and_save_vote_data(df, outpath):
#     df.set_index("date", inplace=True)
#     df.index = pd.to_datetime(df.index)
#     df = df.q_cms
#     df.to_csv(dir_path + outpath + '.csv')

# clean_and_save_vote_data(taylor_park, "taylor_park")
# clean_and_save_vote_data(blue_mesa, "blue_mesa")
# clean_and_save_vote_data(fontenelle, "fontenelle")
# clean_and_save_vote_data(flaming_gorge, "flaming_gorge")
# clean_and_save_vote_data(navajo, "navajo")
# clean_and_save_vote_data(lake_powell, "lake_powell")

taylor_source = df[df['id_source'].isin(['09109000', '09107000', '09107500', '09108000'])].id_gage
taylor_park = sf[sf['id_gage'].isin(taylor_source)]
blue_source = df[df['id_source'].isin(['09128000', '09127000', '09126000', '09124500'])].id_gage
blue_mesa = sf[sf['id_gage'].isin(blue_source)]
# blue_mesa = blue_mesa[blue_mesa['id_gage'] != 15126571863]
fontenelle_source = df[df['id_source'].isin(['09211200', '09209500', '09209400', '09211000'])].id_gage
fontenelle = sf[sf['id_gage'].isin(fontenelle_source)]
# fontenelle = fontenelle[(fontenelle['id_gage'] != 14577771443) & (fontenelle['id_gage'] != 14730895590)]
flaming_source = df[df['id_source'].isin(['09234500', '09229500', '09217000', '09224700'])].id_gage
flaming_gorge = sf[sf['id_gage'].isin(flaming_source)]
navajo_source = df[df['id_source'].isin(['09355500', '09355000', '09346400', '09342500'])].id_gage
navajo = sf[sf['id_gage'].isin(navajo_source)]
lake_source = df[df['id_source'].isin(['09380000', '09379500', '09182400', '09379910'])].id_gage
lake_powell = sf[sf['id_gage'].isin(lake_source)]

dir_path = 'C:/Users/375237/Desktop/CRB-human-impacts/Data/'
def clean_and_save_vote_data(df, outpath):
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = pd.pivot(df, columns='id_gage', values='q_cms').reset_index()
    df.to_csv(dir_path + outpath + '.csv')

# clean_and_save_vote_data(taylor_park, "taylor_park_upstream")
# clean_and_save_vote_data(blue_mesa, "blue_mesa_upstream")
# clean_and_save_vote_data(fontenelle, "fontenelle_upstream")
# clean_and_save_vote_data(flaming_gorge, "flaming_gorge_upstream")
# clean_and_save_vote_data(navajo, "navajo_upstream")
# clean_and_save_vote_data(lake_powell, "lake_powell_upstream")
