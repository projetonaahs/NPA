import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    c_matrix_control = np.load('c_matrix_control.npy')
    c_matrix_adhd = np.load('c_matrix_adhd.npy')
    c_matrix_autism = np.load('c_matrix_autism.npy')

    labels = [0] * len(c_matrix_control) + [1] * len(c_matrix_adhd) + [2] * len(c_matrix_autism)

    # Concatenar as matrizes, não empilhar verticalmente
    X = np.concatenate([c_matrix_control, c_matrix_adhd, c_matrix_autism])

    return X, labels

def preprocess_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler

def train_svm(X_train, y_train):
    clf = SVC(kernel='linear', C=1.0, probability=True)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['control', 'adhd', 'autism'])
    confusion_mat = confusion_matrix(y_test, y_pred)
    return report, confusion_mat

def main():
    # Carregar dados de treinamento
    X_train, labels = load_data()

    # Pré-processar dados de treinamento
    X_train, scaler = preprocess_data(X_train)

    # Dividir dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.2, random_state=42)

    # Treinar modelo
    model = train_svm(X_train, y_train)

    # Utilizando validação cruzada nos dados de treinamento
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f'Acurácia média da validação cruzada nos dados de treinamento: {np.mean(cv_scores):.2f}')

    # Avaliar o modelo nos dados de teste
    report_test, confusion_mat = evaluate_model(model, X_test, y_test)
    print("Classification Report nos dados de teste:")
    print(report_test)

    print("Matriz de Confusão:")
    print(confusion_mat)

    # Carregar e pré-processar a matriz desconhecida
    unknown_matrix = np.load('c_matrix_unknown.npy')

    # Usar o scaler treinado para pré-processar os novos dados
    unknown_matrix_preprocessed = scaler.transform(unknown_matrix)

    # Fazer previsão na matriz desconhecida
    prediction = model.predict(unknown_matrix_preprocessed)

    # Fazer previsão de probabilidades
    probability = model.predict_proba(unknown_matrix_preprocessed)

    print(f'Previsão na matriz desconhecida: {prediction}')
    print(f'Probabilidade: {probability}')

if __name__ == "__main__":
    main()
