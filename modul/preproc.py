import unicodedata
import re
import string

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