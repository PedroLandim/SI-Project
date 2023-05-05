import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


data = pd.read_csv('diabetes.csv')

# split dataset
X = data.iloc[:, 0:8]
y = data.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Define the model: Init K-NN
classifier = KNeighborsClassifier(n_neighbors=11, p=2,metric='euclidean')


# Fit Model
classifier.fit(X_train, y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)
print(y_pred)

# Evaluate Model
conf_matriz = confusion_matrix(y_test, y_pred)
print(conf_matriz)
print(f1_score(y_test, y_pred))

# Final accuracy
print("Precisao do modelo: ", accuracy_score(y_test, y_pred))

for col in data:
  sns.histplot(data=data, x=col, kde=True).set_title(f"Distribuição da variável '{col}'")
  mean = data[col].mean()
  median = data[col].median()
  plt.axvline(median, color='red', linestyle='--')
  plt.axvline(mean, color='yellow', linestyle='--')
  plt.show()

  conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()

class_report = classification_report(y_test, y_pred)
print("Relatório de classificacao:\n", class_report)