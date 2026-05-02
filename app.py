from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = CustomData(
        fixed_acidity        = float(request.form.get("fixed_acidity")),
        volatile_acidity     = float(request.form.get("volatile_acidity")),
        citric_acid          = float(request.form.get("citric_acid")),
        residual_sugar       = float(request.form.get("residual_sugar")),
        chlorides            = float(request.form.get("chlorides")),
        free_sulfur_dioxide  = float(request.form.get("free_sulfur_dioxide")),
        total_sulfur_dioxide = float(request.form.get("total_sulfur_dioxide")),
        density              = float(request.form.get("density")),
        pH                   = float(request.form.get("pH")),
        sulphates            = float(request.form.get("sulphates")),
        alcohol              = float(request.form.get("alcohol")),
    )

    df         = data.get_data_as_dataframe()
    pipeline   = PredictPipeline()
    prediction = pipeline.predict(df)
    result     = round(float(prediction[0]), 2)

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)