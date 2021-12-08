
import sys, os
from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
#from wtforms import Form, StringField, TextAreaField, PasswordField, validators, IntegerField
import csv
import functions as pro
import pandas as pd
import warnings
from flask_sslify import SSLify

app = Flask(__name__)

app.config['Audio_Uploads'] = '/Users/noelalben/Desktop/github/melograph/WebApp/static/Uploads'



@app.route("/", methods=['POST', 'GET'])
def index():
	if request.method == "POST":
		f = request.files['audio_data']
		outname='./static/audio.wav'
		with open(outname, 'wb') as audio:
			f.save(audio)

			
	else:
		return render_template('home.html')

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