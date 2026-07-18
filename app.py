# app.py

import os
import joblib
import gradio as gr

# ==========================================================
# Load the trained model
# ==========================================================
try:
    deployed_dt = joblib.load("Diabetes_Prediction_model.pkl")
except Exception as e:
    print(f"Warning: Model not found or error loading. {e}")
    deployed_dt = None

# ==========================================================
# Prediction Function
# ==========================================================
def predict_diabetes(
    Pregnancies,
    PlasmaGlucose,
    DiastolicBloodPressure,
    TricepsThickness,
    SerumInsulin,
    BMI,
    Age,
):

    values = [
        Pregnancies,
        PlasmaGlucose,
        DiastolicBloodPressure,
        TricepsThickness,
        SerumInsulin,
        BMI,
        Age,
    ]

    # Empty input check
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill in all the input fields."

    # Convert data types
    try:
        Pregnancies = int(Pregnancies)
        PlasmaGlucose = float(PlasmaGlucose)
        DiastolicBloodPressure = float(DiastolicBloodPressure)
        TricepsThickness = float(TricepsThickness)
        SerumInsulin = float(SerumInsulin)
        BMI = float(BMI)
        Age = int(Age)
    except (ValueError, TypeError):
        return "❌ Please enter valid numeric values."

    # Negative value check
    if any(v < 0 for v in [
        Pregnancies, PlasmaGlucose, DiastolicBloodPressure, 
        TricepsThickness, SerumInsulin, BMI, Age
    ]):
        return "❌ Negative values are not allowed."

    # Range validation
    if Pregnancies > 20: return "❌ Pregnancies should be between 0 and 20."
    if PlasmaGlucose > 300: return "❌ Plasma Glucose should be between 0 and 300."
    if DiastolicBloodPressure > 200: return "❌ Blood Pressure should be between 0 and 200."
    if TricepsThickness > 100: return "❌ Triceps Thickness should be between 0 and 100."
    if SerumInsulin > 1000: return "❌ Serum Insulin should be between 0 and 1000."
    if BMI > 70: return "❌ BMI should be between 0 and 70."
    if Age > 120: return "❌ Age should be between 0 and 120."

    if deployed_dt is None:
        return "❌ Model failed to load. Please check your .pkl file."

    try:
        input_data = [[
            Pregnancies,
            PlasmaGlucose,
            DiastolicBloodPressure,
            TricepsThickness,
            SerumInsulin,
            BMI,
            Age,
        ]]

        prediction = deployed_dt.predict(input_data)

        if prediction[0] == 1:
            return (
                "🔴 Prediction Result\n\n"
                "High Risk of Diabetes (Positive)\n\n"
                "Please consult a healthcare professional."
            )
        else:
            return (
                "🟢 Prediction Result\n\n"
                "Low Risk of Diabetes (Negative)\n\n"
                "Maintain a healthy lifestyle."
            )

    except Exception as e:
        return f"❌ Prediction failed.\n\nError: {str(e)}"

# ==========================================================
# Description
# ==========================================================
DESCRIPTION = """
# 🩺 Diabetes Prediction System

This application predicts whether a patient is at **High Risk** or **Low Risk** of Diabetes using a trained **Decision Tree Machine Learning Model**.

---

## 👩‍💻 Developed By
**Manik**

---

## 🏫 College
**Panipat Institute of Engineering & Technology (PIET), Panipat**

---

## 🔗 GitHub Repository
https://github.com/Manik2604/Diabetes-Prediction-Model

---

## 🛠️ Tools & Technologies
* Python
* Machine Learning
* Decision Tree Classifier
* Scikit-learn
* Pandas
* NumPy
* Joblib
* Gradio
* Git & GitHub

---

## 📋 Instructions
1. Clone the repository
   `git clone https://github.com/Manik2604/Diabetes-Prediction-Model.git`
2. Install dependencies
   `pip install gradio scikit-learn pandas numpy joblib`
3. Make sure the file `Diabetes_Prediction_model.pkl` is present in the project folder.
4. Run `python app.py`

---

## 📌 Input Parameters
* Pregnancies
* Plasma Glucose
* Diastolic Blood Pressure
* Triceps Skin Fold Thickness
* Serum Insulin
* Body Mass Index (BMI)
* Age
"""

# ==========================================================
# Interface
# ==========================================================
# --- CODE BLOCK: REMOVED ALLOW_FLAGGING ---
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Plasma Glucose"),
        gr.Number(label="Diastolic Blood Pressure"),
        gr.Number(label="Triceps Skin Fold Thickness"),
        gr.Number(label="Serum Insulin"),
        gr.Number(label="Body Mass Index (BMI)"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(
        label="Assessment Result",
        lines=6,
    ),
    title="🩺 Diabetes Prediction Model",
    description=DESCRIPTION,
)
# ------------------------------------------

# ==========================================================
# Launch
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
