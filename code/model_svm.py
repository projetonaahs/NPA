import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    c_matrix_control = np.load('c_matrix_control.npy')
    c_matrix_adhd = np.load('c_matrix_adhd.npy')
    c_matrix_autism = np.load('c_matrix_autism.npy')

    labels = [0] * len(c_matrix_control) + [1] * len(c_matrix_adhd) + [2] * len(c_matrix_autism)

    X = np.vstack([c_matrix_control, c_matrix_adhd, c_matrix_autism])

    return X, labels

def preprocess_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def split_data(X, labels):
    return train_test_split(X, labels, test_size=0.2, random_state=42)

def train_svm(X_train, y_train):
    clf = SVC(kernel='rbf', C=1.0)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['control', 'adhd', 'autism'])
    return accuracy, report

def predict_class(model, new_data):
    # Pré-processamento semelhante ao utilizado durante o treinamento
    scaler = StandardScaler()
    new_data = scaler.fit_transform(new_data)

    # Classificação usando o modelo SVM treinado
    classe_predita = model.predict(new_data.reshape(1, -1))
    return classe_predita[0]

def main():
    X, labels = load_data()
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X, labels)   

    model = train_svm(X_train, y_train)

    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    #
    new_data = np.load('novo_dado.npy')  
    predicted_class = predict_class(model, new_data)
    print(f"a classe predita para o novo dado é: {predicted_class}")

if __name__ == "__main__":
    main()
