import tensorflow as tf
from tensorflow.compat.v1.keras.layers import CuDNNGRU
from tensorflow.keras.layers import Dense, GRU, Embedding

class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
		super(Decoder, self).__init__()
		self.batch_sz = batch_sz
		self.dec_units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = CuDNNGRU(self.dec_units, return_sequences=True, 
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

def caldec(v, e, enc, ba):
	encoder = Decoder(len(v)+1, e, enc, ba)

	return encoder