import os
import joblib
import gradio as gr

# ==========================================================
# Load Trained Model
# ==========================================================
try:
    deployed_dt = joblib.load("Diabetes_Prediction_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
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

    # Empty input validation
    if any(v is None for v in values):
        return "❌ Please fill in all input fields."

    # Convert data types
    try:
        Pregnancies = int(Pregnancies)
        PlasmaGlucose = float(PlasmaGlucose)
        DiastolicBloodPressure = float(DiastolicBloodPressure)
        TricepsThickness = float(TricepsThickness)
        SerumInsulin = float(SerumInsulin)
        BMI = float(BMI)
        Age = int(Age)

    except ValueError:
        return "❌ Please enter valid numeric values."

    # Negative value validation
    if any(
        x < 0
        for x in [
            Pregnancies,
            PlasmaGlucose,
            DiastolicBloodPressure,
            TricepsThickness,
            SerumInsulin,
            BMI,
            Age,
        ]
    ):
        return "❌ Negative values are not allowed."

    # Range validation
    if Pregnancies > 20:
        return "❌ Pregnancies must be between 0 and 20."

    if PlasmaGlucose > 300:
        return "❌ Plasma Glucose must be between 0 and 300."

    if DiastolicBloodPressure > 200:
        return "❌ Blood Pressure must be between 0 and 200."

    if TricepsThickness > 100:
        return "❌ Triceps Thickness must be between 0 and 100."

    if SerumInsulin > 1000:
        return "❌ Serum Insulin must be between 0 and 1000."

    if BMI > 70:
        return "❌ BMI must be between 0 and 70."

    if Age > 120:
        return "❌ Age must be between 0 and 120."

    if deployed_dt is None:
        return "❌ Model failed to load."

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
        return f"❌ Prediction Failed\n\nError: {e}"


# ==========================================================
# Description
# ==========================================================
DESCRIPTION = """
# 🩺 Diabetes Prediction System

This application predicts whether a patient is at **High Risk** or **Low Risk**
of Diabetes using a trained **Decision Tree Machine Learning Model**.

---

### 👨‍💻 Developed By
**Manik**

### 🏫 College
Panipat Institute of Engineering & Technology (PIET)

### 🛠 Technologies
- Python
- Scikit-learn
- Decision Tree
- Pandas
- NumPy
- Joblib
- Gradio

Enter the patient details and click **Submit**.
"""


# ==========================================================
# Gradio Interface
# ==========================================================
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Plasma Glucose (mg/dL)"),
        gr.Number(label="Diastolic Blood Pressure (mm Hg)"),
        gr.Number(label="Triceps Skin Fold Thickness (mm)"),
        gr.Number(label="Serum Insulin (mu U/mL)"),
        gr.Number(label="Body Mass Index (BMI)"),
        gr.Number(label="Age"),
    ],
    outputs=gr.Textbox(
        label="Assessment Result",
        lines=6,
    ),
    title="🩺 Diabetes Prediction System",
    description=DESCRIPTION,
)


# ==========================================================
# Launch App
# ==========================================================
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
