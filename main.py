# Script principal para execução
# Elaborado por Bernardo Albuquerque Domingues
# https://github.com/beralbdom \ dominguesbernardo@id.uff.br \ bernardo.albuquerque@epe.gov.br
# -------------------------------------------------------------------------------------------------------------------- #

from tratamento import *
from cluster import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------------------ Leitura dos dados e conversão em anomalias ------------------------------------ #

# if __name__ == '__main__':                                                                                              # Execução com hyperthreading
    # ler_torres('Dados', ';')
    # ler_clusters('Dados/Clusters/', ';', '.xlsx')
    # anomalia('clusters', 'Arquivos/ClustersVelMed.csv', 'timestamp')

# Arquivos salvos:
# - Arquivos/TorresVelMed.csv
# - Arquivos/TorresVelMed.xlsx
# - Arquivos/TorresAnomMen.csv
# - Arquivos/TorresAnomMen.xlsx
# - Arquivos/ClustersVelMed.csv
# - Arquivos/ClustersVelMed.xlsx
# - Arquivos/ClustersAnomMen.csv
# - Arquivos/ClustersAnomMen.xlsx


# ------------------------------------------ Regressão dos clusters de torres ---------------------------------------- #

df_torres = pd.read_excel('Arquivos/TorresAnomMen.xlsx').set_index('timestamp')                                         # Lendo o arquivo de anomalias
df_torres.index = pd.to_datetime(df_torres.index).to_period('M')
df_ind = pd.read_csv('Arquivos/Teleconexoes_TCC.csv').set_index('Data')                                                 # Lendo o arquivo de índices
df_ind.index = pd.to_datetime(df_ind.index).to_period('M')

# print(df_ind.columns)

# Leitura dos dados dos clusters e concatenação em um único dataframe
# num_clusters = len(os.listdir('Dados/Clusters'))
# clusters_array = np.zeros(num_clusters, object)
# for i, file in enumerate(os.listdir('Dados/Clusters')):
#     df = pd.read_excel('Dados/Clusters/' + file, usecols = ['datMedicaoConsistida', 
#                         'VelocidadeMédia_mediana']).rename(columns = {'datMedicaoConsistida': 'timestamp'})
#     df = df.set_index('timestamp')
#     df.index = pd.to_datetime(df.index)
#     df = df.resample('ME').mean()
#     clusters_array[i] = df.rename(columns = {'VelocidadeMédia_mediana': file.split('.')[0]})                            # Renomeando a coluna para o nome do arquivo
#     print(f'{file.split('.')[0]} lido!')
# df_clusters = pd.concat(clusters_array, axis = 1)
# df_clusters.to_csv('Arquivos/ClustersVelMed.csv', sep = ';')                                                            # Salvando o dataframe concatenado com a velocidade máxima média mensal de cada cluster

# Cálculo das regressões lineares e geração de gráficos
df_clusters = pd.read_csv('Arquivos/ClustersAnomMen.csv', sep = ';').set_index('timestamp')
df_clusters.index = pd.to_datetime(df_clusters.index).to_period('M')

# regressao(df_torres, df_ind, 'mlinear', graf = True, debug = True, n_comp = 0, nome_saida = '_torres_MAXcomp')
# regressao(df_clusters, df_ind, 'mlinear', graf = True, debug = True, n_comp = 0, nome_saida = '_clusters_MAXcomp')
regressao(df_clusters, df_ind, 'nlinear', graf = True, debug = True, n_comp = 0, nome_saida = '_clusters_MAXcomp')

# dados_regressao = pd.read_excel('Arquivos/Regressao_mlinear_torres_MAXcomp.xlsx')
# dados_regressao = dados_regressao.set_index('Torre')
# dados_regressao_cluster = pd.read_excel('Arquivos/Regressao_mlinear_clusters_MAXcomp.xlsx')
# dados_regressao_cluster = dados_regressao_cluster.set_index('Torre')

# fig, ax = plt.subplots(2, 1, figsize = (6, 6))
# axs = ax.flatten()
# axs[0].hist(dados_regressao['R^2'], bins = 10)
# axs[0].set_title('Regressão Multi-Linear, torres individuais')
# axs[0].set_ylabel('R^2')
# axs[1].hist(dados_regressao_cluster['R^2'], bins = 10)
# axs[1].set_title('Regressão Multi-Linear, clusters')
# axs[1].set_ylabel('R^2')
# plt.tight_layout()
# plt.show()


# ---------------------------------- Cálculo dos clusters e da matriz de correlações --------------------------------- #

# Cálculo dos clusters e da matriz de correlações e geração dos gráficos de dendrograma e de correlações
# X, grupos, df_clust, dado_corr = calc_clusters('Arquivos/Regressao_completo_NOVO.xlsx', 10, 0.5)
# plot_corr(X, grupos, df_clust, dado_corr)

# offsets_reg = pd.read_excel('Arquivos/Regressao_linear.xlsx', sheet_name = 'Sheet1')

# Plotando os gráficos para análise dos offsets de cada índice
# plot_dist_ind(offsets_reg)

# Análise dos offsets de cada índice para cada torre
# torres = offsets_reg['Torre'].unique()
# for torre in torres:
#     analisar_offsets(offsets_reg, torre)