# Funções para tratamento de séries de vento (Vel. méd. mensal e anomalias de velocidade).
# Elaborado por Bernardo Albuquerque Domingues
# https://github.com/beralbdom \ dominguesbernardo@id.uff.br \ bernardo.albuquerque@epe.gov.br
# -------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import os

def porcessar_torres(args):
    pasta, arq, sep, formato, torres = args
    print(f'    Lendo arquivo {arq}...')
    df = pd.read_csv(os.path.join(pasta, arq), 
                     sep = sep, 
                     usecols = ['datMedicaoConsistida', str(torres)], 
                     encoding = 'utf-8')
    
    df = df.rename(columns = {'datMedicaoConsistida': 'timestamp'})
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], format = formato, errors = 'raise')
    df.set_index('timestamp', inplace = True)
    df = df.resample('ME').mean()                                                                                       # Resample dos dados para a média mensal
    df.index = df.index.to_period('M')                                                                                  # Transformação do índice para o formato de período
    return df

def ler_torres(pasta, sep, ext = ('.txt', '.csv'), formato = '%Y-%m-%d %H:%M:%S'):
    arqs = np.array([arq for arq in os.listdir(pasta) if arq.endswith(ext)])
    print(f'Lendo {arqs.size} arquivos na pasta {pasta}...')
    torres = np.array([arq.rsplit('.', 1)[0] for arq in arqs])
    
    with Pool(processes = cpu_count()) as pool:
        dfs_array = pool.map(porcessar_torres, [(pasta, arq, sep, formato, torres[i]) for i, arq in enumerate(arqs)])

    dfs = pd.concat(dfs_array, axis = 1).sort_index()
    dfs.to_csv('Arquivos/TorresVelMed.csv', sep = ';')
    dfs.to_excel('Arquivos/TorresVelMed.xlsx', index = True)

def processar_clusters(args):
    file, pasta, sep, ext = args

    if ext == '.xlsx':
        df = pd.read_excel(pasta + file, 
                        usecols = ['datMedicaoConsistida', 
                                   'VelocidadeMédia_media']).rename(columns = {'datMedicaoConsistida': 'timestamp'})
    elif ext == '.csv':
        df = pd.read_csv(pasta + file, sep = sep, 
                        usecols = ['datMedicaoConsistida', 'VelocidadeMédia_media'], 
                        encoding = 'utf-8').rename(columns = {'datMedicaoConsistida': 'timestamp'})
    
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    df = df.resample('ME').mean()
    df = df.rename(columns = {'VelocidadeMédia_media': file.split('.')[0]})                                             # Renomeando a coluna para o nome do arquivo
    print(f'{file.split(".")[0]} lido!')

    return df

def ler_clusters(pasta, sep, ext):
    num_clusters = len(os.listdir(pasta))
    clusters_array = np.zeros(num_clusters, object)

    with Pool(processes = cpu_count()) as pool:
        clusters_array = pool.map(processar_clusters, [(file, pasta, sep, ext) for file in os.listdir(pasta)])
    
    df_clusters = pd.concat(clusters_array, axis = 1)
    df_clusters.to_csv('Arquivos/ClustersVelMed.csv', sep = ';')                                                        # Salvando o dataframe concatenado com a velocidade máxima média mensal de cada cluster


def anomalia(tipo, arq, timestamp):
    print('\nCalculando anomalias')
    df = pd.read_csv(arq, sep = ';', converters = {timestamp: pd.to_datetime}).set_index(timestamp)                     # Leitura do arquivo de velocidades
    medias_s = df.groupby(df.index.month).mean()                                                                        # Cálculo das médias mensais

    for mes in medias_s.index:
        df.loc[df.index.month == mes] -= medias_s.loc[medias_s.index == mes].values                                     # Cálculo das anomalias

    df = df.rolling(window = 3, min_periods = 1).mean()                                                                 # Suavização das anomalias
    df = df.div(df.std())                                                                                               # Normalização das anomalias (Z-Score)

    df = df.dropna(axis = 1, how = 'all')                                                                               # Remoção de colunas com valores nulos

    if tipo == 'torres':
        df.to_csv('Arquivos/TorresAnomMen.csv', sep = ';')
        df.to_excel('Arquivos/TorresAnomMen.xlsx', index = True)
    elif tipo == 'clusters':
        df.to_csv('Arquivos/ClustersAnomMen.csv', sep = ';')
        df.to_excel('Arquivos/ClustersAnomMen.xlsx', index = True)

    return df