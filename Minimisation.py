import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
from keras import optimizers
import keras.backend as K
from sklearn import metrics
import FuncMinim

# Configure the plot style
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top']       = True
plt.rcParams['ytick.right']     = True

params = {'axes.labelsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
          'figure.figsize': (8,8), 'text.usetex': True,
          'font.family': 'FreeSerif'}
plt.rcParams.update(params)


x = np.linspace(0.1, 5, 0.1)
np.random.shuffle(x)

y = FuncMinim.Functional(x)

print('I\'m here')

fitting_model = Sequential()
fitting_model.add(Dense(1, input_dim=1,activation='linear',trainable=False,
                kernel_initializer='identity',use_bias=False))
fitting_model.add(Dense(32,activation='relu',trainable=True))
fitting_model.add(Dense(64,activation='relu',trainable=True))
fitting_model.add(Dense(128,activation='relu',trainable=True))
fitting_model.add(Dense(256,activation='relu',trainable=True))
fitting_model.add(Dense(1,activation='linear',trainable=True))
#fitting_model.compile(loss='mse', optimizer=optimizers.Adam(decay=1e-3))


fitting_model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-3))
fitting_model.fit(x, y, epochs=200, batch_size=64)
fitting_model.compile(loss='mse', optimizer=optimizers.Adam(lr=5e-4))
fitting_model.fit(x, y, epochs=200, batch_size=64)
fitting_model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4))
fitting_model.fit(x, y, epochs=200, batch_size=64)
fitting_model.compile(loss='mse', optimizer=optimizers.Adam(lr=5e-5))
fitting_model.fit(x, y, epochs=200, batch_size=64)
fitting_model.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-5))
fitting_model.fit(x, y, epochs=200, batch_size=64)

NN_predictions = fitting_model.predict(x)
print('RMSE: {:.3f}'.format(np.sqrt(metrics.mean_squared_error(y,NN_predictions))))
fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0,0,1,1])
ax.minorticks_on()
ax.scatter(x,y,s=2,color='blue')
ax.scatter(x,NN_predictions,s=2,color='red',marker='.',alpha=0.4)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
plt.show()

NN_predictions.min()

x[NN_predictions.argmin()]

fitting_model.save_weights('/Users/marialaurapiscopo/Desktop/CosmoTransitions/Weights/model_weights.h5')

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

minimising_model = Sequential()
minimising_model.add(Dense(1, input_dim=1,activation='linear',trainable=True,
                kernel_initializer='identity',use_bias=False))
minimising_model.add(Dense(32,activation='relu',trainable=False))
minimising_model.add(Dense(64,activation='relu',trainable=False))
minimising_model.add(Dense(128,activation='relu',trainable=False))
minimising_model.add(Dense(256,activation='relu',trainable=False))
minimising_model.add(Dense(1,activation='linear',trainable=False))
minimising_model.compile(loss=identity_loss, optimizer='adam')

minimising_model.load_weights('/Users/marialaurapiscopo/Desktop/CosmoTransitions/Weights/model_weights.h5')

minimising_model.fit(np.ones(5000), y, epochs=50, batch_size=16)

minimising_model.get_weights()[0][0,0]
