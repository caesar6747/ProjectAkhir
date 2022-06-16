import tensorflow as tf
import pickle
import os
from modul.encoder import calenc
from modul.decoder import caldec
from modul.preproc import preprocess_sentence

with open('model-development/dayak/inp_lang.pickle', 'rb') as handle:
	inp_lang = pickle.load(handle)
with open('model-development/dayak/targ_lang.pickle', 'rb') as handle:
	targ_lang = pickle.load(handle)

encoder = calenc(inp_lang.word_index, 32, 256, 2)
decoder = caldec(targ_lang.word_index, 32, 256, 2)

optimizer = tf.keras.optimizers.Adam()
checkpoint_dir = 'model-development/dayak/training_checkpointss'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
								encoder=encoder,
								decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def todayak(sentence):
	sentence = preprocess_sentence(sentence)

	inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
	inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
															maxlen=21,
															padding='post')
	inputs = tf.convert_to_tensor(inputs)
	result = ''
	hidden = [tf.zeros((1, 256))]
	enc_out, enc_hidden = encoder(inputs, hidden)

	dec_hidden = enc_hidden
	dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

	for t in range(21):
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