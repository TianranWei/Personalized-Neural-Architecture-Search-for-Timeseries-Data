from typing import List

from tensorflow import TensorShape
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Subtract, Activation, Dropout
from tensorflow.python.keras.models import Model, Sequential
import numpy as np
from ranker import Ranker

class ComparisonCreator():
    def create_random_comparisons(self, X, number_comparisons=1000):
        indices = []
        for i in range(number_comparisons):
            indices.append(np.random.randint(X.shape[0], size=2))
        return indices

    def create_dataset_from_comparisons(self, X, y, indices):
        # todo make suitable for variing shapes!
        X_1 = np.zeros(shape=(len(indices),X.shape[1]))
        X_2 = np.zeros(shape=(len(indices),X.shape[1]))

        for idx, i in enumerate(indices):
            if y[i[0]] < y[i[1]]: #First one wins
                X_1[idx] = X[i[0]]
                X_2[idx] = X[i[1]]
            else:
                X_1[idx] = X[i[1]]
                X_2[idx] = X[i[0]]
        y = np.ones(shape=(len(indices)))
        return X_1, X_2, y


class RankNet(Ranker):

    def __init__(self,
                 batch_size=10,
                 epochs = 1000,
                 number_comparisons=1000):
        self.batch_size=batch_size
        self.epochs = epochs
        self.number_comparisons = number_comparisons
        self.comparison_creator = ComparisonCreator()

    def fit(self, X_train, y_train):

        indices = self.comparison_creator.create_random_comparisons(X=X, number_comparisons=self.number_comparisons)
        X_1_train, X_2_train, y = self.comparison_creator.create_dataset_from_comparisons(X=X, y=y, indices=indices)

        def _create_base_network(input_dim):
            '''Base network to be shared (eq. to feature extraction).
            '''
            seq = Sequential()
            # seq.add(Dense(input_dim, input_shape=(input_dim,)))
            #seq.add(Dropout(0.1))
            seq.add(Dense(1024, activation='relu'))
            seq.add(Dropout(0.1))
            seq.add(Dense(64, activation='relu'))
            seq.add(Dropout(0.1))
            seq.add(Dense(16))
            seq.add(Dense(1, activation="sigmoid"))
            return seq

        def _create_meta_network(input_dim, base_network):
            input_a = Input(shape=(input_dim,))
            input_b = Input(shape=(input_dim,))

            rel_score = base_network(input_a)
            irr_score = base_network(input_b)

            # subtract scores
            diff = Subtract()([rel_score, irr_score])

            # Pass difference through sigmoid function.
            prob = Activation("sigmoid")(diff)

            # Build model.
            model = Model(inputs=[input_a, input_b], outputs=prob)
            model.compile(optimizer="adadelta", loss="binary_crossentropy") #loss="binary_crossentropy"

            return model


        INPUT_DIM = X_1_train.shape[1]
        base_network = _create_base_network(INPUT_DIM)
        model = _create_meta_network(INPUT_DIM, base_network)
        model.summary()
        #X_1_train, X_2_train, y_train = self._get_X_y(data=self.data)
        model.fit([X_1_train,X_2_train], y)
        self.base_network = base_network

    def predict(self,X) -> List[float]:
        #todo determine output
        score = self.base_network.predict(x=X)
        return score.ravel().tolist()
