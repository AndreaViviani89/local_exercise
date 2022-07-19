from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
# model = pickle.load(open('model_state.pth'))

@app.route('/', methods=['GET'])

def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])

def predict():
    imagefile= request.files['imagefile']
    image_path= './images/' + imagefile.filename
    imagefile.save(image_path)



    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
