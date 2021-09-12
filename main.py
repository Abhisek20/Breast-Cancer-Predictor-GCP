
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

directory = os.path.dirname(__file__)
scale_model = pickle.load(open( 'scale_cancer.pkl', 'rb'))
cancer_model = pickle.load(open('model_cancer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('layout.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # Loads the inputs from UI
    inputs = np.array([float(x) for x in request.form.values()])
    

    try:
        inputs = scale_model.transform(inputs.reshape(1,-1))
    except:
        pass
   

    #print(inputs)
    prediction = cancer_model.predict(inputs)[0]
    #final_features = [np.array(int_features)]



    output_dict = {1:"Malignant Cancer", 0:"Benign Cancer"}
    
    output = output_dict[prediction]

    return render_template('layout.html', prediction_text='Type of cancer predicted is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False )