from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

import string
import re
import pickle
from numpy import array, argmax, random, take
import tensorflow as tf
#from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
import unicodedata
import io
import os
import time

def unicode_to_acii(s):
	return ''.join(c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
	w = unicode_to_acii(w.lower().strip())
	w = re.sub(r"([?.!,¿])", r" \1 ", w)
	w = re.sub(r'[" "]+', " ", w)
	w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
	w = w.strip()
	w = '<start> ' + w + ' <end>'
	return w

class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
		super(Encoder, self).__init__()
		self.batch_sz = batch_sz
		self.enc_units = enc_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = GRU(self.enc_units,
	                       return_sequences=True,
	                       return_state=True,
	                       recurrent_initializer='glorot_uniform')

	def call(self, x, hidden):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state = hidden)
		return output, state

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = GRU(self.dec_units, return_sequences=True, 
							return_state=True, 
							recurrent_initializer='glorot_uniform')
		self.fc = tf.keras.layers.Dense(vocab_size)
		self.W1 = tf.keras.layers.Dense(self.dec_units)
		self.W2 = tf.keras.layers.Dense(self.dec_units)
		self.V = tf.keras.layers.Dense(1)

	def call(self, x, hidden, enc_output):
		hidden_with_time_axis = tf.expand_dims(hidden, 1)
		score = tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
		attention_weights = tf.nn.softmax(self.V(score), axis=1)
		context_vector = attention_weights * enc_output
		context_vector = tf.reduce_sum(context_vector, axis=1)
		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
		output, state = self.gru(x)
		output = tf.reshape(output, (-1, output.shape[2]))
		x = self.fc(output)
		return x, state, attention_weights

	def initialize_hidden_state(self):
		return tf.zeros((self.batch_sz, self.dec_units))

with open('model-development/inp_lang.pickle', 'rb') as handle:
	#print('inp_lang is loaded')
	inp_lang = pickle.load(handle)
with open('model-development/targ_lang.pickle', 'rb') as handle:
	#print('targ_lang is loaded')
	targ_lang = pickle.load(handle)

encoder = Encoder(len(inp_lang.word_index)+1, 500, 256, 64)
decoder = Decoder(len(targ_lang.word_index)+1, 500, 256, 64)

optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = 'model-development/training_checkpointss'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
								encoder=encoder,
								decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def evaluate(sentence):
	sentence = preprocess_sentence(sentence)

	inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
	inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
															maxlen=38,
															padding='post')
	inputs = tf.convert_to_tensor(inputs)

	result = ''

	hidden = [tf.zeros((1, 256))]
	enc_out, enc_hidden = encoder(inputs, hidden)

	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

	for t in range(29):
		predictions, dec_hidden, attention_weights = decoder(dec_input,
															dec_hidden,
															enc_out)

		# storing the attention weights to plot later on
		predicted_id = tf.argmax(predictions[0]).numpy()
		result += targ_lang.index_word[predicted_id] + ' '

		if targ_lang.index_word[predicted_id] == '<end>':
			return result, sentence
		# the predicted ID is fed back into the model
		dec_input = tf.expand_dims([predicted_id], 0)
	return result, sentence

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template("iris-prediction.html")
    elif request.method == 'POST':
        #print(dict(request.form))
        inp_sentence = list(dict(request.form).values())[0]
        print(inp_sentence)
        #iris_features = np.array([float(x) for x in iris_features])
        #model, std_scaler = joblib.load("model-development/iris-classification-using-logistic-regression.pkl")
        #iris_features = std_scaler.transform([iris_features])
        #print(iris_features)
        #result = model.predict(iris_features)
        result, sentences = evaluate(inp_sentence)
        #result = 'sadfas'
        return render_template('iris-prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)