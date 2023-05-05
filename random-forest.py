import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Carregando os dados
data = pd.read_csv("diabetes.csv")

# Definindo as variáveis de entrada e de saída
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Divisão do dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Escalonamento dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criando o modelo Random Forest
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Treinamento do modelo
rfc.fit(X_train, y_train)

# Predição dos resultados
y_pred = rfc.predict(X_test)
y_pred1 = rfc.predict(X_train)

# Avaliando a acurácia do modelo
train_accuracy = accuracy_score(y_train, y_pred1)
accuracy = accuracy_score(y_test, y_pred)
print("Train accuracy:", train_accuracy)
print("accuracy: {:.2f}%".format(accuracy * 100))

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap="Reds", fmt="d")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

# Imprimindo o relatório de classificação
print(classification_report(y_test, y_pred))

# Visualizando a árvore com no máximo 3 níveis
estimator = rfc.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(estimator, feature_names=X.columns, class_names=["0", "1"], filled=True, max_depth=2)
plt.show()
