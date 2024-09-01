from fastapi import FastAPI
# import pickle
import joblib
# import numpy as np
import pandas as pd

# List of all possible symptoms
ALL_SYMPTOMS = ['itching',
 'nodal_skin_eruptions',
 'chills',
 'stomach_pain',
 'muscle_wasting',
 'vomiting',
 'spotting_ urination',
 'fatigue',
 'weight_loss',
 'breathlessness',
 'dark_urine',
 'pain_behind_the_eyes',
 'constipation',
 'abdominal_pain',
 'diarrhoea',
 'yellowing_of_eyes',
 'chest_pain',
 'fast_heart_rate',
 'dizziness',
 'excessive_hunger',
 'slurred_speech',
 'knee_pain',
 'muscle_weakness',
 'unsteadiness',
 'bladder_discomfort',
 'internal_itching',
 'muscle_pain',
 'altered_sensorium',
 'red_spots_over_body',
 'abnormal_menstruation',
 'increased_appetite',
 'lack_of_concentration',
 'receiving_blood_transfusion',
 'stomach_bleeding',
 'distention_of_abdomen',
 'blood_in_sputum',
 'prominent_veins_on_calf',
 'blackheads',
 'small_dents_in_nails',
 'blister']



# model = pickle.load()
model = joblib.load('./model/model.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define a prediction endpoint
@app.post("/predict/")
async def predict(data:dict):
    
    # print(data)
    # print(type(data))
    input_features={}

    for symptom in ALL_SYMPTOMS:
        if symptom in data.keys():
            input_features[symptom]=1
        else:
            input_features[symptom]=0
    l=pd.DataFrame(input_features,index=[0])
    # print(l)
    # print((model))
    prediction = model.predict(l)
    
    # Return the prediction result
    return {"prediction": prediction[0]}

