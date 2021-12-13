
#from wtforms import Form, StringField, TextAreaField, PasswordField, validators, IntegerField

import sys, os
from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
import csv
import functions as pro
import pandas as pd
import warnings
from flask_sslify import SSLify
from flask import send_from_directory
# from .functions import 
from graphs import make_graph

app = Flask(__name__)

app.config['Audio_Uploads'] = './static/Uploads'


@app.route("/", methods=['POST', 'GET'])
def index():
	if request.method == "POST":
		f = request.files['audio_data']
		outname='./static/audio.wav'
		with open(outname, 'wb') as audio:
			f.save(audio)

			
	else:
		return render_template('home.html', data="word")

@app.route('/melograph', methods=['POST', 'GET'])
def melograph_upload():
    if request.method == "POST":
        if request.files:
            Audio = request.files["Audio"]
            print(Audio)
            Audio.save(os.path.join(app.config['Audio_Uploads'], Audio.filename))

            return redirect(request.url)
    else:
        return render_template('melograph.html')

@app.route('/analysis', methods=['POST', 'GET'])
def melograph_analyze():
    if request.method == "POST":
        ##audio functions
        #audio tool audio read
        #HPSS, NCCF, graph
        ## Considering we have a nodes and edges array
        data = make_graph()
        # print(data)
        return render_template('analysis.html', data=data)
    else:
        return render_template('melograph.html')

@app.route("/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory("./", path, as_attachment=False)