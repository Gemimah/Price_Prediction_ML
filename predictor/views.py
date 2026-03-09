import joblib
import pandas as pd
from django.shortcuts import render
from model_generators.clustering.train_cluster import evaluate_clustering_model
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model
from predictor.data_exploration import dataset_exploration, data_exploration, generate_rwanda_district_map
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load models once
regression_model = joblib.load(BASE_DIR / "venv/model_generators/regression/regression_model.pkl")
classification_model = joblib.load(BASE_DIR / "venv/model_generators/classification/classification_model.pkl")
clustering_model = joblib.load(BASE_DIR / "venv/model_generators/clustering/clustering_model.pkl")
clustering_scaler = joblib.load(BASE_DIR / "venv/model_generators/clustering/scaler.pkl")


def data_exploration_view(request):
    df = pd.read_csv(BASE_DIR / "dummy-data/vehicles_ml_dataset.csv")
    map_html = generate_rwanda_district_map(df)  # returns HTML string
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "rwanda_map": map_html,  # pass directly to template
    }
    return render(request, "predictor/index.html", context)


def regression_analysis(request):
    context = {
        "evaluations": evaluate_regression_model()
    }
    if request.method == "POST":
        year = int(request.POST.get("year", 0))
        km = float(request.POST.get("km", 0))
        seats = int(request.POST.get("seats", 0))
        income = float(request.POST.get("income", 0))
        prediction = regression_model.predict([[year, km, seats, income]])[0]
        context["price"] = round(prediction, 2)
    return render(request, "predictor/regression_analysis.html", context)


def classification_analysis(request):
    context = {
        "evaluations": evaluate_classification_model()
    }
    if request.method == "POST":
        year = int(request.POST.get("year", 0))
        km = float(request.POST.get("km", 0))
        seats = int(request.POST.get("seats", 0))
        income = float(request.POST.get("income", 0))
        prediction = classification_model.predict([[year, km, seats, income]])[0]
        context["prediction"] = prediction
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    context = {
        "evaluations": evaluate_clustering_model()
    }
    if request.method == "POST":
        try:
            year = int(request.POST.get("year", 0))
            km = float(request.POST.get("km", 0))
            seats = int(request.POST.get("seats", 0))
            income = float(request.POST.get("income", 0))

            # Step 1: Predict price
            predicted_price = regression_model.predict([[year, km, seats, income]])[0]

            # Step 2: Scale features for clustering
            features_scaled = clustering_scaler.transform([[income, predicted_price]])
            cluster_id = clustering_model.predict(features_scaled)[0]

            # Map cluster to label
            n_clusters = clustering_model.n_clusters
            if n_clusters == 2:
                cluster_mapping = {0: "Standard", 1: "Premium"}
            elif n_clusters == 3:
                cluster_mapping = {0: "Economy", 1: "Standard", 2: "Premium"}
            else:
                cluster_mapping = {i: f"Segment_{i}" for i in range(n_clusters)}

            context.update({
                "prediction": cluster_mapping.get(cluster_id, "Unknown"),
                "price": round(predicted_price, 2)
            })

        except Exception as e:
            context["error"] = str(e)

    return render(request, "predictor/clustering_analysis.html", context)