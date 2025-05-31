from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    assignments_completed = int(request.form["assignments_completed"])
    features = np.array([[study_hours, attendance, assignments_completed]])
    prediction = model.predict(features)
    return render_template("index.html", prediction_text="Predicted Final Score: {:.2f}".format(prediction[0]))

if __name__ == "__main__":
    app.run(debug=True)