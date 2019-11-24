import logging
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import keras

from extract_feature import *
# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

features = GenreFeatureData()

# if all of the preprocessed files do not exist, regenerate them all for self-consistency
is_trainX_exist = os.path.isfile(features.preprossed_trainX)
is_trainY_exist = os.path.isfile(features.preprossed_trainY)
is_valX_exist = os.path.isfile(features.preprossed_valX)
is_valY_exist = os.path.isfile(features.preprossed_trainY)
is_testX_exist = os.path.isfile(features.preprossed_testX)
is_testY_exist = os.path.isfile(features.preprossed_testY)

if (
    is_trainX_exist
    and is_trainY_exist
    and is_valX_exist
    and is_valY_exist
    and is_testX_exist
    and is_testY_exist
):
    print("Preprocessed files exist, deserializing npy files")
    features.load_deserialize_data()
else:
    print("Preprocessing raw audio files")
    features.load_preprocess_data()

print("Training X shape: " , features.train_X.shape)
print("Training Y shape: " , features.train_Y.shape)
print("Validation X shape: " , features.val_X.shape)
print("Validation Y shape: " , features.val_Y.shape)
print("Test X shape: " , features.test_X.shape)
print("Test Y shape: " ,features.test_Y.shape)

input_shape = (features.train_X.shape[1], features.train_X.shape[2])

print("Building RNN model ...")
model = Sequential()
model.add(LSTM(units=128, 
        dropout=0.05, 
        recurrent_dropout=0.35, 
        return_sequences=True, 
        input_shape=input_shape)
        )
model.add(LSTM(units=32,  
        dropout=0.05, 
        recurrent_dropout=0.35, 
        return_sequences=False)
        )
model.add(Dense(units=features.train_Y.shape[1], activation="softmax"))

print("Compiling ...")
optimizer = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()

print("Training ...")
batch_size = 35  # num of training examples per minibatch
num_epochs = 400
history = model.fit(
    features.train_X,
    features.train_Y,
    batch_size=batch_size,
    epochs=num_epochs,
)

print("\nValidating ...")
score, accuracy = model.evaluate(
    features.val_X, features.val_Y, batch_size=batch_size, verbose=1
)
print("Dev loss:  ", score)
print("Dev accuracy:  ", accuracy)


print("\nTesting ...")
score, accuracy = model.evaluate(
    features.test_X, features.test_Y, batch_size=batch_size, verbose=1
)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)
print(history)
print(history.history.keys())
# "Accuracy"
# plt.plot(history.history['accuracy'])

# #plt.plot(history.history['loss'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(['train', 'validation'], loc='upper left')
# plt.savefig('acc_vs_epoch.png')

plt.plot(history.history['loss'])

#plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_vs_epoch.png')
# "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# Creates a HDF5 file 'saved_model.h5'
model_filename = "saved_model.h5"
print("\nSaving model: " + model_filename)
model.save(model_filename)
