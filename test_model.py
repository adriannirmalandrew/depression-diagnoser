import pandas
import numpy

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AvgPool1D, MaxPool1D, Activation, Reshape, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2

#Get training data:
totalX = pandas.read_csv('totalX.csv')
totalX = totalX.apply(pandas.to_numeric, errors='coerce')
#totalX = numpy.resize(totalX, (totalX.shape[0], totalX.shape[1], 1))
totalY = pandas.read_csv('totalY.csv')
totalY = totalY.apply(pandas.to_numeric, errors='coerce')
#totalY = numpy.resize(totalY, (totalY.shape[0], 1, totalY.shape[1]))

#Get testing data:
testX = pandas.read_csv('testX.csv')
testX = testX.apply(pandas.to_numeric, errors='coerce')
testY = pandas.read_csv('testY.csv')
testY = testY.apply(pandas.to_numeric, errors='coerce')

#opt = keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.75, nesterov = True)

#1440 ->C10,1 1331 ->A11 121 ->C7,1
model = Sequential()
model.add(Reshape(input_shape = (None, 1440), target_shape = (1440, 1)))
model.add(BatchNormalization())
model.add(Conv1D(filters = 8, kernel_size = 10, strides = 1, activation = 'relu', activity_regularizer = l2(0.0005)))
model.add(MaxPool1D(pool_size = 11))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv1D(filters = 4, kernel_size = 7, strides = 1, activation = 'relu', activity_regularizer = l2(0.0001)))
model.add(MaxPool1D(pool_size = 5))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Conv1D(filters = 2, kernel_size = 5, strides = 1, activation = 'relu', activity_regularizer = l2(0.0001)))
model.add(MaxPool1D(pool_size = 4))
#model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(4, activation = 'relu'))
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
model.add(Reshape(input_shape = (None, 1, 2), target_shape = (2,)))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#keras.backend.set_value(model.optimizer.learning_rate, 0.01)

#Setting checkpoints:
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
mcp_save = ModelCheckpoint('best_wts.hdf5', save_best_only = True, monitor = 'val_loss', mode = 'min', save_freq = 'epoch')
#reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', mode = 'min', factor = 0.9, patience = 20, verbose = 1, min_delta = 1e-4)

#Helper to prepare row for prediction: 
def prep(df, row):
	return numpy.reshape(df.iloc()[row].to_numpy(), (1,1440))

#Train:
model.fit(x = totalX, y = totalY, validation_split = 0.2, epochs = 1000, batch_size = 50, verbose = 1, callbacks = [mcp_save])

