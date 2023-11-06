import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#carregando as matrizes de conectividade funcional
c_matrix_control = np.load('c_matrix_control.npy')
c_matrix_adhd = np.load('c_matrix_adhd.npy')
c_matrix_autism = np.load('c_matrix_autism.npy')

#atribuind rótulos para os grupos
labels = [0] * len(c_matrix_control) + [1] * len(c_matrix_adhd) + [2] * len(c_matrix_autism)

#combinação de todas as matrizes em um único conjunto de dados
X = np.vstack([c_matrix_control, c_matrix_adhd, c_matrix_autism])

scaler = StandardScaler()
X = scaler.fit_transform(X)


#dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# modelo SVM ---- posteriormente - implementar kernel trick
clf = SVC(kernel='linear', C=1.0)

#treinamento
clf.fit(X_train, y_train)

#prevendo
y_pred = clf.predict(X_test)

#analisando precisão
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['control', 'adhd', 'autism'])

print(f"accuracy: {accuracy:.2f}")
print("classification report:")
print(report)


