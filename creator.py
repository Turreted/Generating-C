import tensorflow as tf
from tensorflow import data
from tensorflow import keras
import numpy as np
import os
import time

datafile = open('bluetooth.c', 'r').read()
chars = sorted(set(datafile))

# create a ditioionary to encode chars with numbers
char_dict = {chars[i]: i for i in range(len(chars))}
char_index = {i: chars[i] for i in range(len(chars))}

# determine length of data unit and number of data units
tensor_length = 100
datasets = len(datafile) // (tensor_length+1)

def encode_text(text: str, char_dict: dict):
    text = [char for char in text]
    for i in range(len(text)):
        text[i] = char_dict[text[i]]
    
    return np.array(text)

training_data = encode_text(datafile, char_dict)

# convert the data to tensor slices
training_data = data.Dataset.from_tensor_slices(training_data)

# convert tensor sequences to desired size (batches)
sequences = training_data.batch(tensor_length+1, drop_remainder=True)

# split dataset into chunks of given and expected text
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

buffer_size = 1000
batch_size = 64

# the map function applies to all batches within the dataset 
sequences = sequences.map(split_input_target)

#shuffle batches within buffer
dataset = sequences.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# vocab size serves as the input size for the initial layer (the input size
# generally seems to refer to the possable inputs)
vocab_size = len(chars)

# yeah I've got no fucking clue what this is
embedding_dim = 256

# same thing. just make this a computer number 
rnn_units = 1024



def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # turns integers into vectors of a fixed size
        # input_dim is the vocab size, and the output is the embedding_dim
        keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        # the RNN part of the neural netowrk. maintains an internal state 
        keras.layers.GRU(embedding_dim, activation='tanh', 
                        recurrent_activation='sigmoid', 
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform',
                        ),
        # we want the output size to be the vocab size, don't need to specify the output shape
        keras.layers.Dense(vocab_size)
    ])

    return model

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)

