import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# pegando o arquivo csv
url = './shopping_trends.csv'
df_inicial = pd.read_csv(url)

# usando o método de Label Encoder para separar as colunas categóricas
# primeiro separando-as
for column in df_inicial.columns:
    if df_inicial[column].dtype == 'object':  # verifica se é uma coluna categórica
        df_inicial[column] = LabelEncoder().fit_transform(df_inicial[column])

X = df_inicial.iloc[:, :-1]     # divisão da parte das Features, menos a última coluna
                                # resolvemos usar todas as colunas com o intuito de conseguir
                                # o máximo de dados passados ao classificador possível

y = df_inicial.iloc[:, -1]      # divisão da parte pro Target, apenas a última coluna

# etapa de divisão para os conjuntos de teste e os conjuntos de treinamento
# teste_size de 0.2, ou basicamente 20%, significa que vamos guardar 20% do dataframe para fazer os testes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# etapa de padronização das Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# etapa de treinamento do Classificador
# vários testes realizados, etapa super importante para poder ser avaliado o desempenho final

indice_inicial_aprendizado = 0.001      # índice inicial de aprendizado
# indice_inicial_aprendizado = 0.010    # quanto maior o número mais rápido o modelo faz as classificações
# indice_inicial_aprendizado = 0.100
# indice_inicial_aprendizado = 0.500

num_max_iter = 100      # número máximo de iterações, basicamente as "épocas"
# num_max_iter = 300
# num_max_iter = 500
# num_max_iter = 1000

# func_ativa = 'relu'       # função de ativação
func_ativa = 'tanh'

clf = MLPClassifier(hidden_layer_sizes=(100, 50, 25), learning_rate_init=indice_inicial_aprendizado, max_iter=num_max_iter, activation=func_ativa, random_state=42)
clf.fit(X_train_scaled, y_train)

# análise de desempenho para os testes futuros
accuracy = clf.score(X_test_scaled, y_test)*100

print('='*30 + '\n')
print(f"Acurácia do modelo: {accuracy:.2f}%\n")
print(f"Índice inicial de aprendizagem: {indice_inicial_aprendizado}\n")
print(f"Número de épocas: {num_max_iter}\n")
print(f"Função de ativação utilizada: {func_ativa}\n")
print("Camadas ocultas: 100, 50, 25\n")
print('='*30)

# plotagem de cada teste para podermos realizar a análise de desempenho
# aqui estamos usando uma matriz de confusão relacionada em cima do nosso target, 'Frequency of Purchases'
# o modelo analisa e retorna uma matriz onde pode ser analisada a quantidade de acertos e erros
# com ela podemos fazer inúmeros cálculos aritméticos, mas focamos na acurácia
# a acurácia pega a quantia de acertos gerais, através do gráfico é possível vê-lá pela diagonal principal
y_pred = clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()

classes = np.unique(y)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

plt.ylabel('Classe Verdadeira')
plt.xlabel('Classe Predita')
plt.tight_layout()

plt.show()