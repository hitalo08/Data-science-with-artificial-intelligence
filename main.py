#matplotlib - Utilizado para gráfico
#Seaborn - Utilizado para Gráfico
#scikit-learn - Inteligência artificial
#---------------------------------------DATA SCIENCE-----------------------------------------
"""
Passo 1 - Entendimento do Desafio
Passo 2 - Entendimento da Àrea/Empresa
Passo 3 - Extração/Obtenção de Dados
Passo 4 - Ajuste de Dados (Tratamento/Limpeza)
Passo 5 - Análise Exploratória
Passo 6 - Modelagem + Algoritmos (IA)
Passo 7 - Interpretação de Resultados
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #pacote inteligência artificial
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#Passo 3 - Extração/Obtenção de Dados
table = pd.read_csv('advertising.csv')
print(table)
#Descobrir a correlação dentro da tabela por meio de gráfico
sns.pairplot(table)  #pairplot cria o gráfico. É utilizado o seaborn para criação
plt.show()  #matplotlib.pyplot.show exibe o gráfico acima.

#Mapa de calor - HeatMap
sns.heatmap(table.corr(), cmap='Wistia', annot=True)  #Exibindo as correlações da tabela.
# O parãmetro cmap define a cor da correlação e o parâmetro annot=True mostra as anotações no mapa de calor.
plt.show()

#Separar os dados em X e Y
y = table['Vendas']  #Prever, calcular
x = table[['TV','Radio','Jornal']]  #Para selecionar mais de um coluna, utilizamos dois colchetes.

#Separar os dados em treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)  #30% para teste.
#train_teste_split utiliza um padrão para as 4 variaveis acima seguindo sempre nesse modelo.

#Inteligência artificial - Regressão Linear e Arvore de Decisão <- são modelos de inteligência artificial

#cria inteligência
modelo_regressaolinear =  LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

#Treino
modelo_regressaolinear.fit(x_treino, y_treino)  #Treinando regressão linear nos dois eixos
modelo_arvoredecisao.fit(x_treino, y_treino)  #Treinando arvore de decisão nos dois eixos.

#Teste de IA
#R² > diz o % que nosso modelo consegue explicar o que aconteceu.
#Cria previsão
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#Compara as previsões
print(f"{r2_score(y_teste, previsao_regressaolinear):.1%}")
print(f"{r2_score(y_teste, previsao_arvoredecisao):.1%}")

#Novas previsões
novos_valores = pd.read_csv('novos.csv')
print(novos_valores)
nova_precisao = modelo_arvoredecisao.predict(novos_valores)
print(nova_precisao)