from datetime import datetime

import pandas as pd
import numpy as np

from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch.nn as nn
import torch.nn.functional as F
import torch

from skorch import NeuralNetClassifier

from models.EntityEmbeddingModel import EntityEmbedding
from util.utilities import label, replace

ex = Experiment("Neural Network with Entity Embedding")
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

def feature_list(record, store_state_data):
    dt = datetime.strptime(record['Date'], '%Y-%m-%d')
    store_index = int(record['Store'])
    year = dt.year
    month = dt.month
    day = dt.day
    day_of_week = int(record['DayOfWeek'])
    try:
        store_open = int(record['Open'])
    except:
        store_open = 1

    promo = int(record['Promo'])

    return [store_open,
            store_index,
            day_of_week,
            promo,
            year,
            month,
            day,
            store_state_data['State'][store_index-1]
            ]


@ex.automain
def my_main():
    train_data = pd.read_csv("data/train.csv")
    store_data = pd.read_csv("data/store.csv")
    state_data = pd.read_csv("data/store_states.csv")

    train_data= train_data[:5000]
    train_data = train_data[train_data['Sales'] != 0]
    train_data = train_data[train_data['Open'] != 0]

    store_state_data = pd.merge(store_data, state_data, left_on='Store', right_on='Store', how='left')

    train_data_X = []
    train_data_y = []


    for index, record in train_data.iterrows():
        fl = feature_list(record,store_state_data)
        train_data_X.append(fl)
        train_data_y.append(int(record['Sales']))
    print("Number of train datapoints: ", len(train_data_y))

    full_X = train_data_X
    full_X = np.array(full_X)
    train_data_X = np.array(train_data_X)


    les = []
    for i in range(train_data_X.shape[1]):
        le = preprocessing.LabelEncoder()
        le.fit(full_X[:, i])
        les.append(le)
        train_data_X[:, i] = le.transform(train_data_X[:, i])

    train_data_X = train_data_X.astype(int)
    train_data_y = np.array(train_data_y)

    train_ratio = 0.9
    num_records = len(train_data_X)
    train_size = int(train_ratio * num_records)




    # Apply entity Embedding to the Categorical variables
    embeddingModel = EntityEmbedding()
    embeddingModel.add('Airline', input_shape=18, output_shape=8)
    embeddingModel.add('AirportFrom', input_shape=293, output_shape=10)
    embeddingModel.add('AirportTo', input_shape=293, output_shape=10)
    embeddingModel.add('DayOfWeek', input_shape=7, output_shape=5)
    embeddingModel.dense('Length', output_shape=1)
    embeddingModel.concatenate()


    X_train, X_ee, y_train, y_ee = train_test_split(train_data_X, train_data_y, test_size=0.1, random_state=44)

    embeddingModel.fit(X_ee, y_ee, X_train, y_train, epochs=10)
    print("Learning embedding completed")
    weights = embeddingModel.get_weight()
    X_train = replace(X_train, weights, embeddingModel.embeddings)
    X_ee = replace(X_ee, weights, embeddingModel.embeddings)
    print("Features replaced with embedding")

    # X_train = X_train.astype('float32').to_numpy()
    # X_ee = X_ee.astype('float32').to_numpy()
    # y = train_data_y.astype('int64').to_numpy()

    mnist_dim = X_train.shape[1]
    hidden_dim = 5
    output_dim = len(np.unique(train_data_y))

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
        lr=0.1)

    print("Training Neural Network")
    net.fit(X_train, y_train)
    print("Training Compeleted")
    y_pred = net.predict(X_ee)

    print(accuracy_score(y_ee, y_pred))
