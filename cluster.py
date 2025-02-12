# Funções para leitura de arquivos de torres, cálculo de anomalias e plotagem de gráficos
# Elaborado por Bernardo Albuquerque Domingues da Silva
# https://github.com/beralbdom \ dominguesbernardo@id.uff.br \ bernardo.albuquerque@epe.gov.br
# -------------------------------------------------------------------------------------------------------------------- #

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error as rmse, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from alive_progress import alive_bar
import scipy.cluster.hierarchy as spc
from matplotlib import pyplot as plt, rc
import pandas as pd
import numpy as np

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html                                                   # Setando parâmetros globais para os gráficos
rc('font', family = 'serif', size = 8)                                                                                  
rc('axes', axisbelow = True, grid = True)
rc('grid', linestyle = '--', alpha = 0.5)


# Alinha a série histórica da torre com cada um dos índices, e depois junta todos os índices alinhados individualmente
# em um único DataFrame. A ideia é remover os dados faltantes mantendo o maior número de amostras válidas possível.
def alinhar_torres_indices(df_torre, df_indices):
    torre = df_torre.columns.to_list()[0]                                                                               # Pegando o nome da torre
    num_indices = len(df_indices.columns)
    torre_ind_alinhados = np.zeros(num_indices, object)
    for i, indice in enumerate(df_indices.columns):
        torre_ind_alinhados[i] = df_torre.join(df_indices[[indice]], how = 'inner').dropna()                            # Alinhando os índices com a torre

    df_combinado = pd.DataFrame()
    for i in range(num_indices):
        torre_ind_alinhados[i] = torre_ind_alinhados[i].drop(columns = torre)                                           # Removendo a coluna da torre e ficando só com os índices
        df_combinado = df_combinado.join(torre_ind_alinhados[i], how = 'outer').dropna(axis = 1)                        # Juntando os índices alinhados com a torre e removendo índices com valores faltantes
    
    df_combinado = df_torre.join(df_combinado, how = 'inner')

    return df_combinado

def analise_pca(df_torre, df_indices, n_comp, reg):                                                                     # https://scikit-learn.org/stable/modules/decomposition.html#pca
    torre = df_torre.columns.to_list()[0]                                                                               # Pegando o nome da torre
    df_combinado = alinhar_torres_indices(df_torre, df_indices)
    X = df_combinado.drop(columns = torre)

    scaler = StandardScaler()                                                                                           # Aplicando Z-Scale normalization
    X_norm = scaler.fit_transform(X)

    # if n_comp == 0: n_comp = min(X.columns.size, X.index.size)                                                          # Se o número de componentes não for especificado, usa o mínimo entre o número de índices e de amostras
    pca = PCA(n_components = n_comp).set_output(transform = 'pandas')
    df_indices_pca = pca.fit_transform(X_norm)
    
    fig, ax = plt.subplots(figsize = (6, 3))
    ax.bar(range(1, n_comp + 1), np.cumsum(pca.explained_variance_ratio_))
    # ax.set_ylabel('Explained Variance Ratio')                                                                         # Não sei o nome disso em portugues
    plt.suptitle(f'Análise de Componentes Principais (PCA) - Torre {torre}')
    ax.set_xticks(range(1, n_comp + 1))
    # ax.set_xticklabels(range(1, n_componentes + 1))
    plt.tight_layout()
    plt.savefig(f'Graficos/{reg}/Analise/torre_{torre}.png', transparent = True)
    plt.close()

    return df_indices_pca, df_combinado


def regressao(df_torres, df_indices, reg, graf = False, debug = False, n_comp = 10, nome_saida = ''):
    torres = df_torres.columns
    dict_reg_ind = {torre: [] for torre in torres}
    dict_reg_ind_melhor = {torre: [] for torre in torres}

    with alive_bar(torres.size, title = f'\n\n  Calculando regressões...', spinner = 'classic', bar = None) as bar:
        for t, torre in enumerate(torres):

            if n_comp != 0: df_indices_pca, df_combinado = analise_pca(df_torres[[torre]], df_indices, n_comp, reg)     # Fazendo PCA torre a torre (colchete duplo resulta num novo dataframe)
            else: 
                df_combinado = alinhar_torres_indices(df_torres[[torre]], df_indices)                                   # Alinhando os índices com a torre
                df_indices_pca = df_combinado.drop(columns = torre)                                                     # Removendo a coluna da torre
            
            indices = df_indices_pca.columns

            if len(df_combinado.index) <= 12 * 3:
                if debug: bar.text(f'Menos de 36 amostras para a torre {torre}. Continuando para a próxima...')
                torre = torres[t + 1]
                continue

            try:
                if reg == 'mlinear':
                    x = df_indices_pca
                    y = df_combinado[torre]

                    modelo = LinearRegression().fit(x, y)
                    y_pred = modelo.predict(x)
                    r2 = r2_score(y, y_pred)
                    dict_reg_ind[torre].append([r2, df_indices_pca.columns.size, df_indices_pca.index.size])            # Salvando os resultados da regressão

                    if graf == True:
                        fig, ax = plt.subplots(figsize = (3, 3))
                        fig.suptitle(f'Torre {torre} - Regressão multi-linear')
                        ax.set_title(f'PCA comp.: {n_comp}, Amostras: {df_indices_pca.index.size}', fontsize = 8)
                        ax.plot([y.min(), y.max()], [y.min(), y.max()], color = 'black', linestyle = '--', linewidth = 1)
                        ax.scatter(y, y_pred, s = 4, c = 'g', alpha = .5)
                        ax.text(0.05, 0.90, (f'$R^2$: {np.round(r2, 3)}'),
                                transform = ax.transAxes, fontsize = 8, 
                                verticalalignment = 'top', bbox = dict(boxstyle = 'round',
                                facecolor = 'none', edgecolor = 'none'))

                # Não-linear
                if reg == 'nlinear':
                    # Não-linear (multi)
                    x = df_indices_pca
                    y = df_combinado[torre]

                    model = RandomForestRegressor()
                    model.fit(x, y)
                    y_pred = model.predict(x)
                    r2 = rmse(y, y_pred)
                    dict_reg_ind[torre].append([r2, df_indices_pca.columns.size, df_indices_pca.index.size])                    # Salvando os resultados da regressão

                    if graf == True:
                            fig, ax = plt.subplots(figsize = (3, 3))
                            fig.suptitle(f'Torre {torre} - Regressão não-linear')
                            ax.set_title(f'PCA comp.: {n_comp}, Amostras: {df_indices_pca.index.size}', fontsize = 8)
                            ax.plot([y.min(), y.max()], 
                                    [y.min(), y.max()], 
                                    color = 'black', linestyle = '--', linewidth = 1)
                            ax.scatter(y, y_pred, s = 4, c = 'g', alpha = .5)
                            ax.text(0.05, 0.90, (f'RMSE: {np.round(r2, 3)}'),
                                    transform = ax.transAxes, fontsize = 8, 
                                    verticalalignment = 'top', bbox = dict(boxstyle = 'round',
                                    facecolor = 'none', edgecolor = 'none'))

                # Linear
                if reg == 'linear':
                    for i, indice in enumerate(indices):
                        offsets = np.arange(-6, 7) 
                        r2_best = -np.inf
                        off_best = 0

                        # Linear
                        x = df_combinado[[indice]]
                        y = df_combinado[[torre]]

                        for offset in offsets:
                            y_shift = y.shift(offset).dropna()                                                          # Fazendo o shift pra comparar com o índice do mês anterior
                            x_shift = x.iloc[:len(y_shift)].dropna()                                                    # Alinhando para compensar o offset

                            # if y_shift.size <= 3 or x_shift.size <= 3: continue                                       # Pulando o loop se não houverem dados
                            model = LinearRegression().fit(x_shift, y_shift)
                            y_pred = model.predict(x_shift)

                            r2 = r2_score(y_shift, y_pred)
                            dict_reg_ind[torre].append([indice, r2, offset])                                            # Salvando os resultados da regressão

                            if r2 > r2_best:
                                y_best = y_pred
                                x_best = x_shift
                                r2_best = r2
                                off_best = offset

                        # print(f'Melhor offset para {indice}: {off_best} (R^2 = {r2_best})')
                        dict_reg_ind_melhor[torre].append([indice, r2_best, off_best])                                    # Salvando os melhores resultados da regressão

                        if graf == True:
                            cols = 8; linhas = int(np.ceil(indices.size / cols))
                            fig, axs = plt.subplots(linhas, cols, figsize = (15, 2 * linhas), sharey = False)
                            fig.suptitle(f'Torre {torre} - Anomalia de velocidade vs. índices')
                            axs = axs.flatten()

                            axs[i].plot(x_best, y_best, c = 'black', alpha = .80, ls = '-', linewidth = 0.5)
                            axs[i].scatter(x, y, s = 4, c = 'g', alpha = .33)
                            axs[i].text(0.05, 0.90, (f'Δ: {off_best}, $R^2$: {np.round(r2_best, 3)}'), 
                                        transform = axs[i].transAxes, fontsize = 8, 
                                        verticalalignment = 'top', bbox = dict(boxstyle = 'round',
                                        facecolor = 'white'))
                
                # Removendo os eixos que não foram utilizados
                if graf == True:
                    if reg == 'linear':
                        for j in range(i + 1, len(axs)):
                            fig.delaxes(axs[j])

                    # print('    Gerando gráficos para a torre', torre + '...')
                    plt.tight_layout()
                    plt.savefig(f'Graficos/{reg}/torre_{torre}_{reg}.png', transparent = True)
                    plt.close()

            except Exception as erro:
                plt.close()
                if debug: 
                    print(f' -> Erro ao processar a torre {torre}: {erro}')
                    # print(f'      X = \n{x}\n      y = \n{y}')
            bar()

    reg_ind_best = [                                                                                                    # Transformando o dict em uma lista
                [torre] + indice 
                for torre, indices in dict_reg_ind_melhor.items()
                for indice in indices
        ]

    reg_ind = [                                                                                                         # Transformando o dict em uma lista
                [torre] + indice
                for torre, indices in dict_reg_ind.items()
                for indice in indices
        ]

    if reg == 'mlinear':
        reg_ind = pd.DataFrame(reg_ind, columns = ['Torre', 'R^2', 'Índice', 'Amostras'])
        reg_ind.to_excel(f'Arquivos/Regressao_{reg}{nome_saida}.xlsx', index = False)

    if reg == 'linear':
        reg_ind_best = pd.DataFrame(reg_ind_best, columns = ['Torre', 'Índice', 'R^2', 'Offset'])
        reg_ind_best.to_excel(f'Arquivos/Regressao_{reg}{nome_saida}.xlsx', index = False)

        df_lin_pivot = reg_ind_best.pivot_table(index = 'Índice', columns = 'Torre', values = 'R^2')
        df_lin_pivot.to_excel(f'Arquivos/Regressao_Pivotada_{reg}{nome_saida}.xlsx', index = True)
        # matriz_corr = df_lin_pivot.corr()
        # matriz_corr.to_excel(f'Arquivos/Correlacao_{reg}{nome_saida}.xlsx', index = True)
        dado_corr = df_lin_pivot.T
        matriz_corr = dado_corr.corr()
        matriz_corr.to_excel(f'Arquivos/Correlacao_{reg}{nome_saida}.xlsx', index=True)

    if reg == 'nlinear':
        reg_ind = pd.DataFrame(reg_ind, columns = ['Torre', 'RMSE', 'Nº de Índice', 'Nº de amostras'])
        reg_ind.to_excel(f'Arquivos/Regressao_{reg}{nome_saida}.xlsx', index = False)


# def analisar_offsets(df, torre):
#     torre = int(torre)
#     df_torre = df[df['Torre'] == torre]                                                                                 # Filtrando o dataframe para a torre desejada
#     indices = df_torre['Índice'].unique()                                                                               # Pegando os índices únicos da torre
#     offsets = df_torre['Offset'].unique()                                                                               # Pegando os offsets únicos da torre

#     # Inicializando a figura
#     cols = 6
#     linhas = int(np.ceil(len(indices) / cols))
#     fig, axs = plt.subplots(linhas, cols, figsize = (15, 2 * linhas), sharey = False)
#     fig.suptitle(f'R$^2$ vs. offsets para a torre {torre}', fontsize = 16)
#     axs = axs.flatten()                                                                                                 # Transformando o array de eixos em um array unidimensional

#     # Plotando os gráficos da torre para cada índice
#     for i, indice in enumerate(indices):
#         r2 = df_torre[df_torre['Índice'] == indice]['R^2']
#         offsets = df_torre[df_torre['Índice'] == indice]['Offset']
#         axs[i].bar(offsets, r2, color = '#0c274a')
#         axs[i].set_title(indice)
#         axs[i].set_xticks(offsets, labels = offsets, fontsize = 7)

#     # Removendo os eixos que não foram utilizados
#     for j in range(i + 1, len(axs)):
#         fig.delaxes(axs[j])

#     plt.tight_layout()
#     plt.savefig(f'Graficos/Dist/Offsets torre {torre}.svg', transparent = True)
#     plt.close()


def calc_clusters(arq, dist, lim):
    '''
    Função para calcular os clusters de torres a partir de uma matriz de correlações. \\
    Os resultados são salvos em 'Arquivos/Grupos.xlsx'.

    Parâmetros:
    -----------
    arq : str \\
        Caminho do arquivo com a matriz de correlações
    dist : float \\
        Distância máxima para a clusterização
    lim : float \\
        Limite de correlação para a clusterização
    '''

    dado2 = pd.read_excel(arq, sheet_name = 'Compilado', index_col = [0, 1])                                            # Lendo o arquivo com a matriz de correlações
    # dado2 = dado2[dado2['Amostras'] >= 50]                                                                              # Considerando amostras acima de 50

    dado_corr = dado2[dado2.columns[:-2]].T                                                                             # Pegando as colunas de índices e transpondo
    correl = dado_corr.corr().replace(np.nan, 0)                                                                        # Calculando a matriz de correlações
    correl = correl.mask(correl <= lim, 0)

    X = spc.linkage(correl, 'ward')                                                                                     # Calculando a matriz de distâncias
    clusters = spc.fcluster(X, dist, criterion = 'distance')                                                            # Calculando os clusters
    df_clust = pd.DataFrame({'Torre': correl.columns.get_level_values(1), 'Grupo': clusters})                           # Criando um dataframe com os resultados
    # grupos = df_clusters['grupo'].sort_values().unique()
    grupos = np.sort(df_clust['Grupo'].unique())

    return X, grupos, df_clust, dado_corr


# def plot_corr(X, grupos, df_clust, dado_corr, excl = []):
#     '''
#     Função para plotar os gráficos de dendrograma e de correlações para cada grupo de torres. \\
#     Os resultados são salvos em 'Graficos/Grupos/Dendrograma.svg' e 'Graficos/Grupos/Grupo {grupo} Indices.svg'.

#     Parâmetros:
#     -----------
#     X : ndarray \\
#         Matriz de distâncias
#     grupos : list \\
#         Lista com os grupos de torres
#     df_clust : DataFrame \\
#         DataFrame com os grupos de torres
#     dado_corr : DataFrame \\
#         DataFrame com a matriz de correlações
#     excl : list \\
#         Lista com os grupos a serem excluídos. Default = []
#     '''

#     grupos = np.setdiff1d(grupos, excl)                                                                                 # Excluindo os grupos desejados

#     # Plotando o dendrograma
#     fig, ax = plt.subplots(figsize = (25, 10))
#     plt.title('Dendrograma de clusterização', fontsize = 10)
#     plt.xlabel('Torres')
#     plt.ylabel('Distância')
#     spc.dendrogram(X, leaf_rotation = 90, leaf_font_size = 4)
#     plt.axhline(10, )
#     plt.savefig('Graficos/Grupos/Dendrograma.svg', transparent = True)
#     plt.close()

#     # Plotando os gráficos de correlações para cada grupo
#     for grupo in grupos:
#         fig, ax = plt.subplots(1, 3, figsize = (12, 6))
#         colunas_plot = df_clust[df_clust['Grupo'] == grupo]['Torre'].astype(int).tolist()                               # Pegando as torres do grupo desejado e convertendo para int
#         dado_corr_valido = dado_corr.loc[:, (slice(None), colunas_plot)]                                                # Filtrando as torres do grupo desejado
#         dado_corr_valido.plot(ax = ax[0], legend = False)                                                               # Plotando o comportamento das torres

#         ax[0].set_title('Comportamento das torres por índice', fontsize = 10)
#         ax[0].tick_params(axis = 'x')
#         ax[0].set_yticks(np.arange(0, 1.2, 0.2))
#         ax[0].set_xticks(np.arange(len(dado_corr.index)))
#         ax[0].set_xticklabels(dado_corr.index, fontsize = 8, rotation = 90)

#         matriz_corr = dado_corr_valido.corr()                                                                           # Calculando a matriz de correlações do grupo desejado 
#         matriz_sup_ind = np.triu_indices_from(matriz_corr, k = 1)                                                       # Pegando os índices da matriz triangular superior
#         matriz_sup = matriz_corr.values[matriz_sup_ind]                                                                 # Filtrando os valores da matriz triangular superior
#         ax[1].hist(matriz_sup, bins = np.arange(0, 1.1, 0.1), color = '#FAAF17', edgecolor = 'black')
#         ax[1].set_title('Distribuição de correlações entre torres', fontsize = 10)
#         ax[1].tick_params(axis = 'x')
#         ax[1].set_xticks(np.arange(0, 1.1, 0.1))
#         ax[1].set_yticks(ax[1].get_yticks())
#         ax[1].set_yticklabels([str(int(x)) + '%' for x in ax[1].get_yticks() * 100 / len(matriz_sup)])

#         cax = ax[2].matshow(dado_corr_valido, cmap = 'RdYlGn', vmin = 0, vmax = 1)                                      # Plotando a matriz de correlações
#         fig.colorbar(cax, ax = ax[2], fraction = .0566, pad = .025)

#         ax[2].set_xticks(np.arange(len(colunas_plot)))
#         ax[2].set_yticks(np.arange(len(dado_corr_valido.index)))
#         ax[2].set_xticklabels(colunas_plot, rotation = 90, fontsize = 7)
#         ax[2].set_yticklabels(dado_corr_valido.index, fontsize = 7)
#         ax[2].grid(False)
#         ax[2].xaxis.set_ticks_position('bottom')

#         # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html                                       # Adicionando a legenda fora do gráfico
#         handles, labels = ax[0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc = 'lower center', ncol = 5, fontsize = 7, bbox_to_anchor = (0.5, 0), 
#                edgecolor = 'black', frameon = False, fancybox = False)

#         fig.suptitle('Análise do grupo '+ str(grupo), fontsize = 14)
#         plt.tight_layout(rect = [0, .18, 1, 0.96])
#         plt.savefig('Graficos/Grupos/Grupo ' + str(grupo) + ' Indices.png', transparent = False)
#         plt.close()


# def plot_dist_ind(df):
#     '''
#     Função para plotar os gráficos de distribuição de offsets para cada índice. \\
#     Os resultados são salvos em 'Graficos/Dist/Distr. offsets.svg' e 'Graficos/Dist/Offsets otimos torre {torre}.svg'.

#     Parâmetros:
#     -----------
#     df : DataFrame \\
#         DataFrame com os dados de regressão (a partir de Regressao_offsets.xlsx) 
#     '''

#     # Pegando as torres e índices únicos
#     indices = df['Índice'].unique()
#     torres = df['Torre'].unique()
 
#     cols = 6
#     linhas = (len(indices) + cols - 1) // cols
#     fig, axes = plt.subplots(linhas, cols, figsize = (15, 2 * linhas))
#     axes = axes.flatten()

#     for i, indice in enumerate(indices):
#         indice_data = df[df['Índice'] == indice]
#         offset_counts = indice_data['Offset'].value_counts().sort_index()                                               # https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html

#         axes[i].bar(offset_counts.index, offset_counts.values, color = '#0c274a')
#         axes[i].set_xticks(offset_counts.index, labels = offset_counts.index, fontsize = 7)
#         axes[i].set_title(indice)
#         axes[i].set_xlabel('Offsets', fontsize = 6)
#         axes[i].set_ylabel('Ocorrências', fontsize = 6)

#     # Removendo os eixos que não foram utilizados
#     for j in range(i + 1, len(axes)):
#         fig.delaxes(axes[j])

#     fig.suptitle(f'Distribuição de offsets para cada índice', fontsize = 16)
#     plt.tight_layout()
#     plt.savefig('Graficos/Dist/Distr. offsets.svg', transparent = True)