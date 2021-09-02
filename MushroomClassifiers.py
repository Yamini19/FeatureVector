import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split, cross_validate
import matplotlib.pyplot as plt

# Models for classification task
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from models.EntityEmbeddingModel import EntityEmbedding
from util.utilities import label, replace
from sacred import Experiment
from sacred.observers import MongoObserver


ex = Experiment("Airline Delay Prediction-- Entity Embedding input")
ex.observers.append(MongoObserver())

@ex.config
def my_config():
    oneHotEncoding_input = False
    EntityEmbedding_input = True
    size = 5000


@ex.capture
def getdata(size):
    data = pd.read_csv("data/airline.csv")
    data = data.drop(['Flight', 'Time'], axis=1)
    data = data[:size]
    return data



def getOneHotEncoding(X):
        X = pd.get_dummies(X, columns=['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek'])
        return X


def getEntityEmbedding(X, y, data):
        # Apply Label Encoding to the Categorical features
        label(data, 'Airline')
        label(data, 'AirportFrom')
        label(data, 'AirportTo')

        # Apply entity Embedding to the Categorical variables
        embeddingModel = EntityEmbedding()
        embeddingModel.add('Airline', input_shape=18, output_shape=8)
        embeddingModel.add('AirportFrom', input_shape=293, output_shape=10)
        embeddingModel.add('AirportTo', input_shape=293, output_shape=10)
        embeddingModel.add('DayOfWeek', input_shape=7, output_shape=5)
        embeddingModel.dense('Length', output_shape=1)
        embeddingModel.concatenate()

        X = data.loc[:, ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Length']]

        X['Airline'] = X['Airline'].astype(float, errors='raise')
        X['AirportFrom'] = X['AirportFrom'].astype(np.float32)
        X['AirportTo'] = X['AirportTo'].astype(np.float32)
        X['DayOfWeek'] = X['DayOfWeek'].astype(np.float32)

        X_train, X_ee, y_train, y_ee = train_test_split(X, y, test_size=0.25, random_state=44)

        embeddingModel.fit(X_ee, y_ee, X_train, y_train, epochs=12)

        weights = embeddingModel.get_weight()
        X = replace(X, weights, embeddingModel.embeddings)
        return X


@ex.capture
def FeatureEncoder(X, y, data, oneHotEncoding_input, EntityEmbedding_input):
    if oneHotEncoding_input:
        X = getOneHotEncoding(X)
    elif EntityEmbedding_input:
        X = getEntityEmbedding(X, y, data)

    return X


@ex.capture
def plotGraph(results_accuracy, results_recall, results_precision, results_f1, names,
              oneHotEncoding_input, EntityEmbedding_input):
    # boxplot algorithm comparison
    fig = plt.figure()
    if oneHotEncoding_input:
        fig.suptitle('Algorithm Comparison-- OneHotEncoding (Recall)')
    elif EntityEmbedding_input:
        fig.suptitle('Algorithm Comparison-- Entity Embedding (Recall)')
    ax = fig.add_subplot(111)
    plt.boxplot(results_recall)
    ax.set_xticklabels(names)
    plt.ylabel("Recall")
    plt.show()
    if oneHotEncoding_input:
        fig.savefig('plots/OneHotEncoding_Recall.png')
    elif EntityEmbedding_input:
        fig.savefig('plots/EntityEmbedding_Recall.png')

    # boxplot algorithm comparison
    fig = plt.figure()
    if oneHotEncoding_input:
        fig.suptitle('Algorithm Comparison-- OneHotEncoding (Accuracy)')
    elif EntityEmbedding_input:
        fig.suptitle('Algorithm Comparison-- Entity Embedding (Accuracy)')
    ax = fig.add_subplot(111)
    plt.boxplot(results_accuracy)
    ax.set_xticklabels(names)
    plt.ylabel("Accuracy")
    plt.show()
    if oneHotEncoding_input:
        fig.savefig('plots/OneHotEncoding_Accuracy.png')
    elif EntityEmbedding_input:
        fig.savefig('plots/EntityEmbedding_Accuracy.png')

    # boxplot algorithm comparison
    fig = plt.figure()
    if oneHotEncoding_input:
        fig.suptitle('Algorithm Comparison-- OneHotEncoding (Precision)')
    elif EntityEmbedding_input:
        fig.suptitle('Algorithm Comparison-- Entity Embedding (Precision)')
    ax = fig.add_subplot(111)
    plt.boxplot(results_precision)
    ax.set_xticklabels(names)
    plt.ylabel("Precision")
    plt.show()
    if oneHotEncoding_input:
        fig.savefig('plots/OneHotEncoding_Precision.png')
    elif EntityEmbedding_input:
        fig.savefig('plots/EntityEmbedding_Precision.png')

    # boxplot algorithm comparison
    fig = plt.figure()
    if oneHotEncoding_input:
        fig.suptitle('Algorithm Comparison-- OneHotEncoding (f1 score)')
    elif EntityEmbedding_input:
        fig.suptitle('Algorithm Comparison-- Entity Embedding (f1 score)')

    ax = fig.add_subplot(111)
    plt.boxplot(results_f1)
    ax.set_xticklabels(names)
    plt.ylabel("f1 score")
    plt.show()
    if oneHotEncoding_input:
        fig.savefig('plots/OneHotEncoding_f1.png')
    elif EntityEmbedding_input:
        fig.savefig('plots/EntityEmbedding_f1.png')


@ex.automain
def my_main():
    data = getdata()
    X = data.loc[:, ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Length']]
    y = data.Delay
    #Get encoded Feature vectors for categorical data
    X= FeatureEncoder(X, y , data)


    models = [('LR', LogisticRegression(max_iter=1000)), ('DTC', DecisionTreeClassifier()),
              ('KNN', KNeighborsClassifier()),
              ('NB', GaussianNB()),
              # ('SVM', SVC(kernel="linear")),
              ('RFC', RandomForestClassifier(max_depth=5, n_estimators=10)),
              ('GBC', GradientBoostingClassifier())]

    results_accuracy = []
    results_recall = []
    results_precision = []
    results_f1 = []
    names = []
    seed = 7
    scoring = ['accuracy', 'recall', 'f1', 'precision']
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_validate(model, X, y, cv=kfold, scoring=scoring)
        results_recall.append(cv_results['test_recall'])
        results_accuracy.append(cv_results['test_accuracy'])
        results_precision.append(cv_results['test_precision'])
        results_f1.append(cv_results['test_f1'])

        names.append(name)
        msg = "%s: Accuracy: %f Recall: %f Precision: %f f1: %f" % (name, cv_results['test_accuracy'].mean(),
                                                                    cv_results['test_recall'].mean(),
                                                                    cv_results['test_precision'].mean(),
                                                                    cv_results['test_f1'].mean())
        print(msg)

    plotGraph(results_accuracy, results_recall, results_precision, results_f1, names)