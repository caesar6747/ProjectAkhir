from flask import Flask, render_template, request, jsonify
import numpy as np
import string
from numpy import array, argmax, random, take
from modul.trans_dayak import todayak
from modul.trans_jawa import tojawa
import io
import os
import time


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def iris_prediction():
    if request.method == 'GET':
        return render_template("index.html")
    elif request.method == 'POST':
        print(dict(request.form))
        inp_sentence = list(dict(request.form).values())[0]
        lang = list(dict(request.form).values())[1]
        print(lang)
        if lang == 'jaw':
            result, sentences = tojawa(inp_sentence)
        elif lang == 'day':
            result, sentences = todayak(inp_sentence)
        return render_template('index.html', result=result, before=sentences)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)