import gradio as gr
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model
model = joblib.load('titanic_voting_model.pkl')

# 2. Define the prediction function
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Create a DataFrame for the input
    # Note: Column names must match the training data
    data = pd.DataFrame({
        'Pclass': [int(pclass)],
        'Sex': [sex],
        'Age': [float(age)],
        'SibSp': [int(sibsp)],
        'Parch': [int(parch)],
        'Fare': [float(fare)],
        'Embarked': [embarked]
    })
    
    # Predict
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1] # Probability of surviving (class 1)
    
    result = "Survived" if prediction == 1 else "Did Not Survive"
    return result, f"{probability:.2%}"

# 3. Create Gradio Interface
iface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown(choices=["1", "2", "3"], label="Pclass (Ticket Class)"),
        gr.Radio(choices=["male", "female"], label="Sex"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Age"),
        gr.Slider(minimum=0, maximum=8, step=1, label="SibSp (Siblings/Spouses)"),
        gr.Slider(minimum=0, maximum=6, step=1, label="Parch (Parents/Children)"),
        gr.Number(label="Fare", value=32.2),
        gr.Radio(choices=["S", "C", "Q"], label="Embarked (Port of Embarkation)")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Survival Probability")
    ],
    title="Titanic Survival Predictor",
    description="Enter passenger details to predict if they would survive the Titanic disaster.",
    theme="default"
)

# 4. Launch the app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0")
