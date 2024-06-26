from hate.entity.config_entity import ModelTrainerConfig
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, Input, SpatialDropout1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from hate.constants import *

class ModelArchitecture:
    def __init__(self):
        pass

    def get_model(self):
        model = Sequential()
        model.add(Embedding(MAX_WORDS, 10, input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss=LOSS, optimizer=RMSprop(), metrics=METRICS)

        return model