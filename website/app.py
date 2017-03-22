import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, jsonify

from flask_util_js import FlaskUtilJs

from werkzeug import secure_filename
import json
import subprocess

from image_category_predict import Imagepredict

ob = Imagepredict()

UPLOAD_FOLDER = 'Uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])



app = Flask(__name__)
fujs = FlaskUtilJs(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.context_processor
def inject_fujs():
    return dict(fujs=fujs)
    

def allowed_file(filename):
	return '.' in filename and filename.split('.',1)[1] in ALLOWED_EXTENSIONS


def save_to_disk(file):
	if (file and allowed_file(file.filename)):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return filename
	else:
		return None


@app.route('/')
def index():
	return render_template('index.html')




@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if (request.method == 'POST'):
		image_file = request.files['image_file']
		imageFileName = save_to_disk(image_file)
		
		# ob = Imagepredict()
		res = ob.predict_image(os.path.join(app.config['UPLOAD_FOLDER'], imageFileName))
		return jsonify(file_name=imageFileName, res=res)
	else:
		print "Request"


@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == "__main__":
	app.run(debug=True)


