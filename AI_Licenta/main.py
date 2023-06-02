from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import accuracy_score

from Model import *
from GenerateDataSet import *
from sklearn.model_selection import RepeatedKFold
import pandas as pd

if __name__ == '__main__':
    model = create_model(15, 5)
    generateDataSet(10000, 15)

    df = pd.read_csv("question_scores_dataset.csv")

    X = df.iloc[:, :15].to_numpy()
    y = df.iloc[:, 15:].to_numpy()

    print(X.shape)
    print(y.shape)

    # X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)

    results = list()
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        model.fit(X_train, y_train, verbose=1, epochs=100)
        yhat = model.predict(X_test)
        yhat = yhat.round()
        acc = accuracy_score(y_test, yhat)
        print('>%.3f' % acc)
        results.append(acc)

    print(results)
    model.save("model.h5")
