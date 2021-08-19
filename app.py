#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
from helper import detect, load_model, get_prediction_text
import helper

#Initialize the Flask app
app = Flask(__name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates')

# global model
model = load_model()

@app.route('/')
def root():
    helper.reset_prediction_text()
    return render_template('gray.html')

@app.route('/home')
def home():
    helper.reset_prediction_text()
    return render_template('gray.html')

@app.route('/OD')
def OD():
    helper.reset_prediction_text()
    return render_template('gray-consent.html')

@app.route('/video_feed')
def video_feed():
    # global model
    return Response(detect(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction_text', methods=['GET'])
def get_prediction_text():
    return Response(helper.get_prediction_text(), content_type='text/plain')

@app.route('/reset_prediction_text')
def reset_prediction_text():
    helper.reset_prediction_text()
    return "nothing"

if __name__ == "__main__":
    # global model
    # model = load_model()
    app.run(debug=True, use_reloader=False)