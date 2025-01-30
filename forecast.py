# Forecast de dados anemométricos usando como hiperparâmetros índices climatológicos
# Elaborado por Bernardo Albuquerque Domingues
# https://github.com/beralbdom \ dominguesbernardo@id.uff.br \ bernardo.albuquerque@epe.gov.br
# -------------------------------------------------------------------------------------------------------------------- #

# https://stackoverflow.com/questions/63517126/any-way-to-predict-monthly-time-series-with-scikit-learn-in-python
# estudar se metodo para avaliar o erro do forecast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tratamento as io

from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect

torre_vel, torre_num = io.ler_torres('Dados', ';', arq = '10.csv')
torre_anom = io.anomalia(torre_vel, 'MS', 'timestamp')
indices = pd.read_csv('teleconexoes_TCC.csv', converters = {'Data': pd.to_datetime}).set_index('Data')
agregado = torre_anom[0].join(indices, how = 'inner').drop_duplicates()

agregado = agregado.asfreq('MS', fill_value = agregado.mean())
data_train = agregado[: -36]
data_test = agregado[-36 :]
data = data_train['velocidade']

start_date = agregado.index.min()
end_date = agregado.index.max()
date_range = pd.date_range(start = start_date, end = end_date, freq = 'MS')
print(f'\nDataset: {date_range[0]} - {date_range[-1]} (n = {len(date_range)})')
print(f'Treino : {data_train.index.min()} - {data_train.index.max()} (n = {len(data_train)})')
print(f'Teste  : {data_test.index.min()} - {data_test.index.max()} (n = {len(data_test)})\n')

# -------------------------------------------------------------------------------------------------------------------- #

# kpss_test = KPSS(data_train['velocidade'])
# print(kpss_test.summary().as_text())
# fig, axs = plt.subplots(2, 1, figsize = (6, 3))
# axs[0].plot(data, label = 'original')
# plt.show()

forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(n_estimators = 500, 
                                                      criterion = 'squared_error', 
                                                      max_features = 1, 
                                                      n_jobs = -1, 
                                                      verbose = 1),
                    lags      = 6)
forecaster.fit(y = data,
               exog = data_train[['AMM', 'AO', 'TNA']])

prev = forecaster.predict(steps = 36,
                          exog = data_test[['AMM', 'AO', 'TNA']])
# prev = pd.Series(data = prev, index = data_test.index)

fig, ax = plt.subplots(1, 1, figsize = (6, 3))
ax.plot(data_train['velocidade'], label = 'treino')
ax.plot(data_test['velocidade'], label = 'teste')
ax.plot(prev, label = 'previsao')
ax.legend(); plt.show()