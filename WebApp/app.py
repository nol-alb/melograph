
import sys, os
from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, IntegerField
import csv
import functions as pro
import pandas as pd
import warnings
from flask_sslify import SSLify

app = Flask(__name__)


@app.route('/')
def melograph_home():

    return render_template('home.html')

@app.route('/melograph')
def melograph_upload():
    return render_template('melograph.html')