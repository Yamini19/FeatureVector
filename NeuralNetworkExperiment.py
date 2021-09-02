import pandas as pd
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.nn.functional as F
import torch

from skorch import NeuralNetClassifier
from util.utilities import label, replace

ex = Experiment("Neural Network with One hot Encoding")
ex.observers.append(MongoObserver())

@ex.config
def my_config():
    size = 10000

@ex.capture
def getdata(size):
    data = pd.read_csv("data/airlines.csv")
    data = data.drop(['Flight', 'Time'], axis=1)
    data = data[:size]
    return data


@ex.automain
def my_main():
    data=getdata()

    X = data.loc[:, ['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek', 'Length']]
    y = data.Delay

    X = pd.get_dummies(X, columns=['Airline', 'AirportFrom', 'AirportTo', 'DayOfWeek'])

    X = X.astype('float32').to_numpy()
    y = y.astype('int64').to_numpy()


    # # Apply entity Embedding to the Categorical variables
    # embeddingModel = EntityEmbedding()
    # embeddingModel.add('Airline', input_shape=18, output_shape=8)
    # embeddingModel.add('AirportFrom', input_shape=293, output_shape=10)
    # embeddingModel.add('AirportTo', input_shape=293, output_shape=10)
    # embeddingModel.add('DayOfWeek', input_shape=7, output_shape=5)
    # embeddingModel.dense('Length', output_shape=1)
    # embeddingModel.concatenate()
    #
    #
    #
    # X['Airline'] = X['Airline'].astype(float, errors='raise')
    # X['AirportFrom'] = X['AirportFrom'].astype(np.float32)
    # X['AirportTo'] = X['AirportTo'].astype(np.float32)
    # X['DayOfWeek'] = X['DayOfWeek'].astype(np.float32)

    X_train, X_ee, y_train, y_ee = train_test_split(X, y, test_size=0.25, random_state=44)

    # embeddingModel.fit(X_ee, y_ee, X_train, y_train, epochs=12)
    # print("Learning embedding completed")
    # weights = embeddingModel.get_weight()
    # X_train = replace(X_train, weights, embeddingModel.embeddings)
    # X_ee = replace(X_ee, weights, embeddingModel.embeddings)
    # print("Features replaced with embedding")

    mnist_dim = X.shape[1]
    hidden_dim = 5
    output_dim = len(np.unique(y))


    class ClassifierModule(nn.Module):
        def __init__(
                self,
                input_dim=mnist_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.5,
        ):
            super(ClassifierModule, self).__init__()
            self.dropout = nn.Dropout(dropout)
            self.hidden1 = nn.Linear(mnist_dim, 100)
            self.hidden2 = nn.Linear(100, 50)
            self.output = nn.Linear(50, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden1(X))
            X = F.relu(self.hidden2(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=1)
            return X


    torch.manual_seed(0)

    net = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=20,
        lr= 0.1)

    print("Training Neural Network")
    net.fit(X_train, y_train)
    print("Training Compeleted")
    y_pred = net.predict(X_ee)

    print(accuracy_score(y_ee, y_pred))