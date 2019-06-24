py
import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import talos as ta
from talos.model.normalizers import lr_normalizer
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, RepeatVector
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import plot_model
from keras.optimizers import Adam, Nadam, RMSprop
from keras.activations import softmax, relu, elu
from keras.losses import categorical_crossentropy, logcosh
from keras.callbacks import EarlyStopping
from keras import regularizers
import sklearn as sk
from sklearn import preprocessing
log1 = pd.read_csv(r"C:\Users\edfim\OneDrive\Documentos\Psynary\RCPsych conference\Master.csv")
trial = log1.loc[log1['test'] <1]
X = trial.drop(labels=['username', 'rem', 'rem2', 'rem3', 'rem4'], axis=1).values
y = trial.rem.values #change to 2 and 3 for different experiment
seed= 1003
np.random.seed(seed)
tf.set_random_seed(seed)
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)
test_index = np.array(list(set(range(len(X))) - set (train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]
train_index2 = np.random.choice(len(X), round(len(X) * 1), replace=False)
train_xx = X[train_index2]
train_yy = y[train_index2]
scaler = sk.preprocessing.StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

encoding_dim = 22
input_dim = Input(shape=(105,))
encoded = Dense(encoding_dim, activation='relu')(input_dim)
decoded = Dense(105, activation='sigmoid')(encoded)

encoder = Model(input_dim, encoded)
autoencoder = Model(input_dim, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

history = autoencoder.fit(train_X, train_X,
    epochs=2000,
    batch_size=1000,
    shuffle=True,
    validation_data=(test_X, test_X))


x_train = encoder.predict (train_xx)
y_train = train_yy
x_test = encoder.predict (test_X)
true_test_x = log1.loc[log1['test']>0]
xx = true_test_x.drop(labels=['username', 'rem', 'rem2', 'rem3', 'rem4'], axis=1).values
yy = true_test_x.rem.values
test_index2 = np.random.choice(len(xx), round(len(xx) * 1), replace=False)
test_xx = xx[test_index2]
test_yy = yy[test_index2]
xx_test = encoder.predict (test_xx)
yy_test = test_yy



def simple_bpad_ttrad_model (x_train, y_train, x_test, y_test, params):
    model = Sequential()
    model.add(Dense(params['first_neuron'],
                   input_dim=x_train.shape[1],
                   activation='relu'))
    keras.layers.Dropout(params['dropout'])
    model.add(Dense(1,
                   activation=params['last_activation']))
    model.compile(optimizer=params['optimizer'](lr=lr_normalizer(params['lr'], params['optimizer'])),
                   loss=params['losses'],
                   metrics=['acc'])
    earlystop = EarlyStopping(monitor='val_acc', patience=4, mode='auto')
    out = model.fit(x_train, y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    verbose=0,
                    validation_data=[x_test, y_test])
    return out, model

p = {'lr':(0.8, 1.2, 3),
     'first_neuron':[42, 55, 64, 73, 80],
     'hidden_layers':[0, 1, 2],
     'batch_size': [200, 300, 400],
     'epochs': [300],
     'dropout': (0, 0.1, 3),
     'optimizer': [Adam, Nadam, RMSprop],
     'losses': ['binary_crossentropy'],
     'last_activation': ['sigmoid']}

h = ta.Scan(x_train, y_train, params=p,
            model=simple_bpad_ttrad_model)



=====================================================================================================================-
encoding_dim = 26
input_dim = Input(shape=(105,))

encoded = Dense(encoding_dim, activation='relu',)(input_dim)
decoded = Dense(105, activation='sigmoid')(encoded)
autoencoder = Model(input_dim, decoded)
encoder = Model(input_dim, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))





plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Autoencoder loss history')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Test loss'], loc='upper left')
plt.show()


input_dim = Input(shape=(105,))
encoded = Dense(52, activation='relu')(input_dim)
#encoded = Dense(26, activation='relu')(encoded)
#encoded = Dense(13, activation='relu')(encoded)
#decoded = Dense(26, activation='relu')(encoded)

decoded = Dense(52, activation='relu')(encoded)
decoded = Dense(105, activation='sigmoid')(decoded)