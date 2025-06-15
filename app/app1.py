from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/trynow')
def trynow():
    return render_template('trynow.html')

@app.route('/traditional')
def traditional():
    return render_template('traditional.html')

@app.route('/western')
def western():
    return render_template('western.html')

@app.route('/casual')
def casual():
    return render_template('casual.html')

if __name__ == '__main__':
    app.run(debug=True)