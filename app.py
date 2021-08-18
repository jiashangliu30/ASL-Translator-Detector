#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
from helper import detect, load_model

#Initialize the Flask app
app = Flask(__name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates')

# global model
model = load_model()

@app.route('/')
def root():
    return render_template('gray.html')

@app.route('/home')
def home():
    return render_template('gray.html')

@app.route('/OD')
def OD():
    return render_template('gray-consent.html')

@app.route('/video_feed')
def video_feed():
    # global model
    return Response(detect(model), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # global model
    # model = load_model()
    app.run(debug=True, use_reloader=False)