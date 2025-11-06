from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
success_model = pickle.load(open("models/success_model.pkl", "rb"))
funding_model = pickle.load(open("models/funding_model.pkl", "rb"))
year_model = pickle.load(open("models/year_model.pkl", "rb"))


# -------------------------------
# Utility: Safe float conversion
# -------------------------------
def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# -------------------------------
# Success label logic
# -------------------------------
def label_success(score, low_thresh=0.06, high_thresh=0.33):
    if score >= high_thresh:
        return "High"
    elif score >= low_thresh:
        return "Medium"
    else:
        return "Low"


# -------------------------------
# Year model simulation logic
# -------------------------------
def simulate_growth(profile, year):
    years_since_start = year - 2025
    return {
        "funding": profile["funding"] * (1.2**years_since_start),
        "employee_log": np.log1p(profile["employee_count"] * (1.15**years_since_start)),
        "sector": profile["sector"],
        "city_encoded": profile["city_encoded"],
        "state_encoded": profile["state_encoded"],
    }


def simulate_success_year_only(profile, model, high_thresh=0.210):
    for year in range(2025, 2036):
        features = simulate_growth(profile, year)
        X_input = pd.DataFrame(
            [features],
            columns=[
                "funding",
                "employee_log",
                "sector",
                "city_encoded",
                "state_encoded",
            ],
        )
        score = model.predict(X_input)[0]
        if score >= high_thresh:
            return year
    return None


# -------------------------------
# Feature extraction functions
# -------------------------------
def extract_success_features(form):
    sector = int(safe_float(form.get("sector")))
    stage = int(safe_float(form.get("stage")))
    funding_round = int(safe_float(form.get("funding_round")))
    funding_raw = safe_float(form.get("funding_range"))
    funding_log = np.log1p(funding_raw)
    city_encoded = int(safe_float(form.get("district")))
    state_encoded = int(safe_float(form.get("state")))

    return np.array(
        [
            [
                sector,
                stage,
                funding_round,
                funding_log,
                city_encoded,
                state_encoded,
            ]
        ]
    )


def extract_funding_features(form):
    year = safe_float(form.get("year"))
    sector = int(safe_float(form.get("sector")))
    stage = int(safe_float(form.get("stage")))
    funding = safe_float(form.get("funding_range"))
    funding_round = int(safe_float(form.get("funding_round")))
    employee_count = safe_float(form.get("employee_count"))
    investor_count = safe_float(form.get("investor_count"))
    city_encoded = int(safe_float(form.get("district")))
    state_encoded = int(safe_float(form.get("state")))

    employee_log = np.log1p(employee_count)
    investors_log = np.log1p(investor_count)

    return np.array(
        [
            [
                year,
                sector,
                stage,
                funding,
                funding_round,
                employee_log,
                investors_log,
                city_encoded,
                state_encoded,
            ]
        ]
    )


# -------------------------------
# Routes
# -------------------------------
@app.route("/report")
def intro_report():
    return render_template("report.html")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/existing", methods=["GET"])
def existing():
    return render_template("existing.html")


@app.route("/existing/result", methods=["POST"])
def existing_result():
    try:
        success_features = extract_success_features(request.form)
        funding_features = extract_funding_features(request.form)

        # Predict success score and label
        success_score = success_model.predict(success_features)[0]
        success_label = label_success(success_score)

        # Predict funding
        funding_amount = funding_model.predict(funding_features)[0]

        return render_template(
            "result1.html",
            success_label=success_label,
            funding_amount=round(funding_amount, 2),
        )

    except Exception as e:
        print("Prediction error:", e)
        return (
            f"<h2>Prediction failed</h2><p>{str(e)}</p><a href='/'>Return to Home</a>"
        )


@app.route("/future", methods=["GET"])
def future():
    return render_template("future.html")


@app.route("/future/result", methods=["POST"])
def future_result():
    profile = {
        "funding": safe_float(request.form.get("funding_range")),
        "employee_count": int(safe_float(request.form.get("employee_count"))),
        "sector": int(safe_float(request.form.get("sector"))),
        "city_encoded": int(safe_float(request.form.get("district"))),
        "state_encoded": int(safe_float(request.form.get("state"))),
    }
    predicted_year = simulate_success_year_only(profile, year_model)
    return render_template("result2.html", predicted_year=predicted_year)


# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
