import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data():

    #training files 
    c_matrix_adhd = np.load('c_matrix_adhd.npy')
    c_matrix_autism = np.load('c_matrix_autism.npy')
    #training files 

    labels = [0] * len(c_matrix_adhd) + [1] * len(c_matrix_autism)

    X = np.concatenate([c_matrix_adhd, c_matrix_autism])

    return X, labels

def preprocess_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, scaler

def train_svm(X_train, y_train):
    clf = SVC(kernel='rbf', C=1.0, probability=True)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['adhd', 'autism'])
    confusion_mat = confusion_matrix(y_test, y_pred)
    return report, confusion_mat

def format_probability(probability):
    formatted_probs = []
    for prob in probability:
        formatted_probs.append([f"{p:.2%}" for p in prob])

    return formatted_probs

def main():
    X_train, labels = load_data()

    X_train, scaler = preprocess_data(X_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.2, random_state=42)

    model = train_svm(X_train, y_train)
 
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    print(f'cross validation accuracy: {np.mean(cv_scores):.2f}')

    report_test, confusion_mat = evaluate_model(model, X_test, y_test)
    print("report (trainment):")
    print(report_test)

    print("confusion matrix:")
    print(confusion_mat)

    unknown_matrix = np.load('c_matrix_unknown.npy') #non-diag.
    unknown_matrix_preprocessed = scaler.transform(unknown_matrix)

    prediction = model.predict(unknown_matrix_preprocessed)
    probability = model.predict_proba(unknown_matrix_preprocessed)

    formatted_probability = format_probability(probability)

    probability = model.predict_proba(unknown_matrix_preprocessed)
    formatted_probability = format_probability(probability)
    print("probability:")
    for row in formatted_probability:
        print(row)

    print(f'prev. unknown matrix: {prediction}')
    

    unique_elements, counts = np.unique(prediction, return_counts=True)
    for element, count in zip(unique_elements, counts):
        if count >= 19:
            if element == 0:
                print("major matches: Attention Deficit Hyperactivity Disorder (ADHD)")
            elif element == 1:
                print("major matches: Autism Spectrum Disorder (ASD)")

if __name__ == "__main__":
    main()
