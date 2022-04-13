import numpy as np
import pandas as pd
import os
import re
import time
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential, model_from_json
from keras import regularizers
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Activation, AveragePooling1D, Dropout, CuDNNGRU
from keras.callbacks import *
import tensorflow as tf
import math


###############################################
### Change your directory, idk if you need it #
#os.chdir('D:/Brandon/NDSC 2019/data')        #
###############################################


### Set up the train, validation, and test data
train_df = pd.read_csv('train.csv')
train_df = train_df.sample(frac=1.) # this shuffles the df

train_ratio = 0.7; validation_ratio = 0.15

total_size = len(train_df) 

train_size = int(train_ratio * total_size)
validation_size = int(validation_ratio * total_size)

train_data = train_df[:train_size]
validation_data = train_df[train_size:train_size+validation_size]
test_data = train_df[validation_size:]


### Set up embedding
embeddings_index = {}
f = open('custom_glove_100d.txt')

for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

### Transforming each batch into numpy arrays

def text_to_array(text):
    empty_embed = np.zeros(400)
    text = text[:-1].split()[:100]
    embeds = [embeddings_index.get(x, empty_embed) for x in text]
    embeds+=[empty_embed]* (100-len(embeds))
    return np.array(embeds)

batch_size = 128     ### Change batch size here
i = 0
texts = train_data.iloc[i*batch_size:(i+1)*batch_size, 1]
text_arr = np.array([text_to_array(text) for text in texts])
batch_labels = np.array(train_data["Category"][i*batch_size:(i+1)*batch_size])
batch_targets = np.zeros((batch_size, 58))
batch_targets[np.arange(batch_size), batch_labels] = 1



### Create batch generator
def batch_gen(train_df):
    n_batches = math.floor(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            batch_labels = np.array(train_df["Category"][i*batch_size:(i+1)*batch_size])
            yield text_arr, batch_labels

			
			
####################			
### Create model ###
####################

model = Sequential() 
model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True),
                        input_shape=(100, 400)))
model.add(AveragePooling1D(data_format='channels_last'))
model.add(Bidirectional(CuDNNLSTM(256))) 
model.add(Dense(256))
model.add(Activation(tf.nn.leaky_relu))
model.add(Dropout(0.6)) 
model.add(Dense(58))
model.add(Activation('softmax'))


#Saving best accuaracy & loss
mcp_acc = ModelCheckpoint('bestAccmodel', monitor='val_acc', save_best_only=True, save_weights_only=True, mode='max', period=1)
mcp_loss = ModelCheckpoint('bestLossmodel', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)

#Reducing LR on Peak
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1, mode='min')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

mg = batch_gen(train_data)
history = model.fit_generator(mg, epochs=30,
                    steps_per_epoch=math.ceil(train_size/batch_size),
                    validation_data=batch_gen(validation_data),
                    validation_steps=math.ceil(validation_size/batch_size),
		            callbacks = [mcp_acc, mcp_loss, reduce_lr],
                    verbose=True)

model_json = model.to_json()
with open("modelXII.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelXII.h5")
print("Saved model to disk")
with open('modelXII.json', 'r') as json_file:
    bestModel = model_from_json(json_file.read(), custom_objects={'leaky_relu': tf.nn.leaky_relu})
bestModel.load_weights('bestAccmodelXII.h5f')

batch_size = 256
def batch_gen(test_df):
    n_batches = math.ceil(len(test_df) / batch_size)
    for i in range(n_batches):
        texts = test_df.iloc[i*batch_size:(i+1)*batch_size, 1]
        text_arr = np.array([text_to_array(text) for text in texts])
        yield text_arr

test_df = pd.read_csv("test.csv")

test_df["Supercategory"] = test_df["image_path"].str[0]
supercats = np.array(test_df["Supercategory"])
supercat_dict = {
    "b" : np.array([1]*17 + [0]*14 + [0]*27),
    "f" : np.array([0]*17 + [1]*14 + [0]*27),
    "m" : np.array([0]*17 + [0]*14 + [1]*27)
}

all_preds = []
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(bestModel.predict(x))

y_te = [np.argmax(pred) for pred,supercat in zip(all_preds,supercats)]

submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category":y_te})
submit_df.to_csv("submissionXIIAcc.csv", index = False)

model.load_weights('bestLossmodelXII.h5f')
for x in tqdm(batch_gen(test_df)):
    all_preds.extend(bestModel.predict(x))
y_te = [np.argmax(pred) for pred,supercat in zip(all_preds,supercats)]

submit_df = pd.DataFrame({"itemid": test_df["itemid"], "Category":y_te})
submit_df.to_csv("submissionXIILoss.csv", index = False)


import matplotlib.pyplot as plt

history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss,'bo', label = 'training loss')

plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.clf()
plt.plot(epochs, acc, 'bo', label = 'training acc')
plt.plot(epochs, val_acc, 'b', label = 'validation acc')
plt.title('training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.show()

