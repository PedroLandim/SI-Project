import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Carrega o conjunto de dados
data = pd.read_csv('diabetes.csv')


# Separa as características e a variável dependente
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Normaliza as características
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# Divide o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.1, random_state=0)

# Treina o modelo Naive Bayes Gaussiano
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Faz previsões no conjunto de teste
y_pred = gnb.predict(X_test)


# Calcula a precisão do modelo
accuracy = gnb.score(X_test, y_test)
print("Precisão do modelo: {:.2f}%".format(accuracy * 100))


# Imprime o relatório de classificação
class_report = classification_report(y_test, y_pred)
print("Relatório de classificação:\n", class_report)

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()