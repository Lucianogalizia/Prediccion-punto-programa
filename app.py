from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Load models and original mappings (with spaces in keys) ---
model1 = joblib.load("models/model1.pkl")
orig_mapping1 = joblib.load("models/mapping1.pkl")
model2 = joblib.load("models/model2.pkl")
orig_mapping2 = joblib.load("models/mapping2.pkl")

# --- Normalize mapping keys to use underscores instead of spaces ---
mapping1 = {k.replace(" ", "_"): v for k, v in orig_mapping1.items()}
mapping2 = {k.replace(" ", "_"): v for k, v in orig_mapping2.items()}

# --- Define FEATURES lists with underscore keys matching our mappings ---
FEATURES1 = [
    "rig_name",
    "loc_fed_lease_no",
    "field_name",
    "Eventos_Normalizados",
    "MANIOBRAS_NORMALIZADAS",
    "GEO_LATITUDE",
    "GEO_LONGITUDE"
]
FEATURES2 = [
    "rig_name",
    "loc_fed_lease_no",
    "field_name",
    "Eventos_Normalizados",
    "MANIOBRAS_NORMALIZADAS",
    "pickup_weight",
    "GEO_LATITUDE",
    "GEO_LONGITUDE"
]

# --- Target log-transformer ---
log_tr = FunctionTransformer(np.log1p, inverse_func=np.expm1)

# --- Load well coordinates ---
coords_df = pd.read_excel("data/coordenadas1.xlsx", dtype={"POZO": str})
for c in ["GEO_LATITUDE", "GEO_LONGITUDE"]:
    coords_df[c] = (
        coords_df[c]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# --- Apply frequency encoding to new data ---
def apply_fe_new(df_new, mapping):
    df = df_new.copy()
    for col, m in mapping.items():
        df[col] = df[col].map(m).fillna(0)
    return df

# --- Predict a single maneuver's duration ---
def predict_single(inputs: dict) -> float:
    mani = inputs["MANIOBRAS_NORMALIZADAS"]
    # Choose model based on whether it comes from dataset2
    if mani in mapping2["MANIOBRAS_NORMALIZADAS"].index:
        mdl, mp, feats = model2, mapping2, FEATURES2
    else:
        mdl, mp, feats = model1, mapping1, FEATURES1

    df = pd.DataFrame([inputs])[feats]
    df_enc = apply_fe_new(df, mp)
    dmat = xgb.DMatrix(df_enc)
    pred_t = mdl.predict(dmat)
    return float(log_tr.inverse_transform(pred_t.reshape(-1, 1))[0, 0])

# --- Home route ---
@app.route("/")
def home():
    # Build pick lists from mappings and coordinates
    rig_names   = list(mapping1["rig_name"].index)
    lease_nos   = list(mapping1["loc_fed_lease_no"].index)
    field_names = list(mapping1["field_name"].index)
    eventos     = list(mapping1["Eventos_Normalizados"].index)
    # unified maneuvers list
    maniobras   = sorted(set(
        mapping1["MANIOBRAS_NORMALIZADAS"].index.tolist()
        + mapping2["MANIOBRAS_NORMALIZADAS"].index.tolist()
    ))
    pozos       = coords_df["POZO"].tolist()
    maniobras_2 = list(mapping2["MANIOBRAS_NORMALIZADAS"].index)

    return render_template(
        "form.html",
        rig_names=rig_names,
        lease_nos=lease_nos,
        field_names=field_names,
        eventos=eventos,
        maniobras=maniobras,
        pozos=pozos,
        maniobras_2=maniobras_2
    )

# --- Coordinates lookup ---
@app.route("/coords/<pozo>")
def coords(pozo):
    row = coords_df[coords_df["POZO"] == pozo].iloc[0]
    return jsonify(lat=row["GEO_LATITUDE"], lon=row["GEO_LONGITUDE"])

# --- Batch prediction route ---
@app.route("/predict", methods=["POST"])
def do_predict():
    # Fixed inputs
    rig   = request.form["rig_name"]
    lease = request.form["loc_fed_lease_no"]
    field = request.form["field_name"]
    evento = request.form["Eventos_Normalizados"]  # form uses underscore
    lat = float(request.form["GEO_LATITUDE"])
    lon = float(request.form["GEO_LONGITUDE"])

    # Selected maneuvers (multiple)
    maniobras = request.form.getlist("MANIOBRAS_NORMALIZADAS")

    # Hidden JSON with pick weights per maneuver
    weights = {}
    if "pickup_weights" in request.form:
        weights = json.loads(request.form["pickup_weights"])

    results = []
    total = 0.0

    for m in maniobras:
        inp = {
            "rig_name":                rig,
            "loc_fed_lease_no":        lease,
            "field_name":              field,
            "Eventos_Normalizados":    evento,
            "MANIOBRAS_NORMALIZADAS":  m,
            "pickup_weight":           float(weights.get(m, 0)),
            "GEO_LATITUDE":            lat,
            "GEO_LONGITUDE":           lon
        }
        dur = predict_single(inp)
        results.append({"maniobra": m, "duracion": round(dur, 2)})
        total += dur

    return render_template("results.html", results=results, total=round(total, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


