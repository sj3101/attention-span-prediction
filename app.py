from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load(r"E:\Education\college\internship\ds_ibm\Attention_Span_Prediction_Project\model training\model_dump.pkl")
scaler_y = joblib.load(r"E:\Education\college\internship\ds_ibm\Attention_Span_Prediction_Project\model training\scaler_y.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        user_input = {
            "Content_Type": request.form["Content_Type"],
            "Content_Length": float(request.form["Content_Length"]),
            "Scroll_Depth": float(request.form["Scroll_Depth"]),
            "Interaction_Count": int(request.form["Interaction_Count"]),
            "Time_Of_Day": request.form["Time_Of_Day"],
            "Day_Of_Week": request.form["Day_Of_Week"],
            "Device_Type": request.form["Device_Type"],
            "Platform_Group": request.form["Platform_Group"]
        }

        input_df = pd.DataFrame([user_input])
        print("DEBUG - Input columns:", input_df.columns.tolist())

        prediction_scaled = model.predict(input_df)
        prediction_original = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

        return render_template("result.html", prediction=round(prediction_original[0][0], 2))

if __name__ == '__main__':
    app.run(debug=True)