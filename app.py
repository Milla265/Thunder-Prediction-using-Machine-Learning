import numpy as np
import pickle
from flask import Flask,request,render_template


app=Flask(__name__)

model=pickle.load(open('thunder_data.pkl','rb'))

@app.route('/')


def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
    int_features=[float(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    
    output=float(prediction)
    
    return render_template('index.html',prediction_text='The number of Thunderstorm for the Month is = {}'.format(output))

if __name__ == '__main__':
    app.run()
