#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from pydantic import BaseModel
import dill
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the saved user model data
with open('api_ey.pkl', 'rb') as file:
    user_model_data = dill.load(file)

# Extract the components from the loaded data
model = user_model_data['model']
scaler = user_model_data['scaler']
label_encoders = user_model_data['label_encoders']
feature_columns = user_model_data['feature_columns']
loaded_models = user_model_data['loaded_models']
loaded_label_encoders = user_model_data['loaded_label_encoders']
predict_with_model_loaded = user_model_data['predict_with_model_loaded']
calculate_impact = user_model_data['calculate_impact']
provide_recommendations = user_model_data['provide_recommendations']
severity_percentages = user_model_data['severity_percentages']
impact_hypertension = user_model_data['impact_hypertension']
impact_diabetes = user_model_data['impact_diabetes']
impact_heart_disease = user_model_data['impact_heart_disease']
impact_cancer = user_model_data['impact_cancer']

# Define Pydantic model for input data
class InputData(BaseModel):
    Age: int
    BMI: float
    Gender: str
    Smoking_Frequency: int
    Alcohol_Consumption_Frequency: int
    Physical_Activity_Level: int
    Sleep_Duration: int
    Disease_Management_Hypertension: int
    Disease_Management_Diabetes: int
    Disease_Management_Heart_Disease: int
    Disease_Management_Cancer: int

# Define a route to accept user input and return predictions
@app.post("/predict/")
async def predict(input_data: InputData):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Apply label encoding to the categorical columns
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])

    # Scale the input data
    scaled_input = scaler.transform(input_df[feature_columns])

    # Make the main prediction (e.g., age prediction)
    predicted_data = model.predict(scaled_input)

    # Run additional predictions (e.g., disease severity predictions)
    target_columns = [
        'Disease_Severity_Hypertension', 'Disease_Severity_Diabetes',
        'Disease_Severity_Heart_Disease', 'Disease_Severity_Cancer'
    ]
    result_df = predict_with_model_loaded(input_data.dict(), loaded_models, loaded_label_encoders, target_columns)

    # Convert predictions to DataFrames for easy manipulation
    predicted_data_df = pd.DataFrame(predicted_data, columns=['Predicted_Age'])
    result_df = pd.DataFrame(result_df)

    # Combine the required columns (BMI, Alcohol_Consumption_Frequency, Smoking_Frequency) with predictions
    input_df_selected = input_df[['BMI', 'Alcohol_Consumption_Frequency', 'Smoking_Frequency']]
    combined_df = pd.concat([input_df_selected, predicted_data_df, result_df], axis=1)

    # Convert the DataFrame to a dictionary for returning as JSON
    result_dict = combined_df.to_dict(orient='records')

    return {"predictions": result_dict}



# In[ ]:




