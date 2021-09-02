import pandas as pd
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Activation, Reshape, Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding


class EntityEmbedding:
    def __init__(self):
        self.input_model = []
        self.output_model = []
        self.features = []
        self.embeddings = []

    def add(self, feature, input_shape, output_shape):
        self.features.append(feature)
        self.embeddings.append(feature)
        input_model = Input(shape=(1,), name=(feature + '_input'))
        output_model = Embedding(input_shape, output_shape, name=(feature + '_embedding'))(input_model)
        output_model = Reshape(target_shape=(output_shape,))(output_model)
        self.input_model.append(input_model)
        self.output_model.append(output_model)

    def dense(self, feature, output_shape):
        self.features.append(feature)
        input_model = Input(shape=(1,), name=(feature + '_input'))
        output_model = Dense(output_shape, name=(feature + '_dense'))(input_model)
        self.input_model.append(input_model)
        self.output_model.append(output_model)

    def concatenate(self):
        output_model = Concatenate()(self.output_model)
        output_model = Dense(1000, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(500, kernel_initializer="uniform")(output_model)
        output_model = Activation('relu')(output_model)
        output_model = Dense(1)(output_model)
        output_model = Activation('sigmoid')(output_model)
        self.model = KerasModel(inputs=self.input_model, outputs=output_model)
        self.model.compile(loss='mean_absolute_error', optimizer='adam')

    def split_features(self, X):
        X_list = {}
        for feature in self.features:
            X_list[feature + '_input'] = X[[feature]]
        return X_list

    def fit(self, X_train, y_train, X_test, y_test, epochs=12, batch_size=70):
        self.X_test = X_test
        self.model.fit(self.split_features(X_train), y_train,
                       validation_data=(self.split_features(X_test), y_test),
                       epochs=epochs,
                       batch_size=batch_size)

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        pred = self.model.predict(self.split_features(X))
        return pred

    def get_weight(self):
        weights = {}
        for feature in self.embeddings:
            w = self.model.get_layer(feature + '_embedding').get_weights()[0]
            columns = []
            for i in range(w.shape[1]):
                columns.append(feature + '_' + str(i))
            w = pd.DataFrame(w, columns=columns)
            w.index.names = [feature]
            weights[feature] = w
        return weights


