from flask import Flask, render_template, request
import pickle, joblib
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
ct = joblib.load('feature_values')

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/pred')
def predict():
    return render_template("index.html")

@app.route('/out', methods =["POST"])
def output():
    discharge_disposition_id = request.form["discharge_disposition_id"]
    admission_source_id = request.form["admission_source_id"]
    time_in_hospital = request.form["time_in_hospital"]
    num_medications = request.form["num_medications"]
    number_emergency = request.form["number_emergency"]
    number_inpatient = request.form["number_inpatient"]
    diag_1 = request.form["diag_1"]
    diag_2 = request.form["diag_2"]
    max_glu_serum = request.form["max_glu_serum"]
    glimepiride = request.form["glimepiride"]
    diabetesMed = request.form["diabetesMed"]
    
    if max_glu_serum == '>200' or max_glu_serum=='>300':
        max_glu_serum=1
    elif max_glu_serum=='Norm':
        max_glu_serum=0
    else:
        max_glu_serum=-99
        
    if glimepiride == 'No':
        glimepiride = 0
    else:
        glimepiride=1
    
    
    
    data = [[discharge_disposition_id,admission_source_id,time_in_hospital,
             num_medications, number_emergency, number_inpatient,diag_1, diag_2,
             max_glu_serum, glimepiride, diabetesMed]]
    
    
    feature_cols = ['discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 
                    'num_medications','number_emergency', 'number_inpatient', 
                    'diag_1', 'diag_2','max_glu_serum', 'glimepiride', 'diabetesMed']

    pred = model.predict(ct.transform(pd.DataFrame(data,columns=feature_cols)))
    pred = pred[0]
    if pred:
        return render_template("output.html",y="This patient will be readmitted ")
    else:
        return render_template("output.html",y="This patient will not be readmitted")

if __name__ == '__main__':
    app.run(debug = True)
    