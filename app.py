from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Carga de modelos y mappings ---
model1 = joblib.load("models/model1.pkl")
mapping1 = joblib.load("models/mapping1.pkl")
model2 = joblib.load("models/model2.pkl")
mapping2 = joblib.load("models/mapping2.pkl")

# --- Listas de features, con los mismos nombres que usan los mappings ---
FEATURES1 = [
    "rig_name",
    "loc_fed_lease_no",
    "field_name",
    "Eventos Normalizados",
    "MANIOBRAS NORMALIZADAS",
    "GEO_LATITUDE",
    "GEO_LONGITUDE"
]
FEATURES2 = [
    "rig_name",
    "loc_fed_lease_no",
    "field_name",
    "Eventos Normalizados",
    "MANIOBRAS NORMALIZADAS",
    "pickup_weight",
    "GEO_LATITUDE",
    "GEO_LONGITUDE"
]

# --- Transformer de target log1p ↔ expm1 ---
log_tr = FunctionTransformer(np.log1p, inverse_func=np.expm1)

# --- Carga del Excel de coordenadas de pozos ---
coords_df = pd.read_excel("data/coordenadas1.xlsx", dtype={"POZO": str})
for c in ["GEO_LATITUDE", "GEO_LONGITUDE"]:
    coords_df[c] = (
        coords_df[c]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# --- Frequency‐encoding para nuevos datos ---
def apply_fe_new(X_new, mapping):
    X = X_new.copy()
    for col, m in mapping.items():
        X[col] = X[col].map(m).fillna(0)
    return X

# --- Predicción de una sola maniobra ---
def predict_single(inputs: dict) -> float:
    mani = inputs["MANIOBRAS NORMALIZADAS"]
    if mani in mapping2["MANIOBRAS NORMALIZADAS"].index:
        mdl, mp, feats = model2, mapping2, FEATURES2
    else:
        mdl, mp, feats = model1, mapping1, FEATURES1

    df = pd.DataFrame([inputs])[feats]
    df_enc = apply_fe_new(df, mp)
    dmat = xgb.DMatrix(df_enc)
    pred_t = mdl.predict(dmat)
    # deshago log1p
    return float(log_tr.inverse_transform(pred_t.reshape(-1,1))[0,0])

# --- Página principal ---
@app.route("/")
def home():
    return render_template(
        "form.html",
        rig_names=list(mapping1["rig_name"].index),
        lease_nos=list(mapping1["loc_fed_lease_no"].index),
        field_names=list(mapping1["field_name"].index),
        eventos=list(mapping1["Eventos Normalizados"].index),
        maniobras=sorted(
            set(mapping1["MANIOBRAS NORMALIZADAS"].index.tolist() +
                mapping2["MANIOBRAS NORMALIZADAS"].index.tolist())
        ),
        pozos=coords_df["POZO"].tolist(),
        maniobras_2=list(mapping2["MANIOBRAS NORMALIZADAS"].index)
    )

# --- Endpoint para autocompletar lat/lon al elegir pozo ---
@app.route("/coords/<pozo>")
def coords(pozo):
    row = coords_df[coords_df["POZO"] == pozo].iloc[0]
    return jsonify(lat=row["GEO_LATITUDE"], lon=row["GEO_LONGITUDE"])

# --- Manejo de la predicción de lote ---
@app.route("/predict", methods=["POST"])
def do_predict():
    # Extraigo inputs fijos
    rig = request.form["rig_name"]
    lease = request.form["loc_fed_lease_no"]
    field = request.form["field_name"]
    evento = request.form["Eventos_Normalizados"]  # form name tiene guión bajo
    lat = float(request.form["GEO_LATITUDE"])
    lon = float(request.form["GEO_LONGITUDE"])

    # Renombro la clave para que coincida con FEATURES
    evento_key = evento
    # Extraigo lista de maniobras seleccionadas
    maniobras = request.form.getlist("MANIOBRAS_NORMALIZADAS")

    # Cargo JSON oculto con los pesos por maniobra
    weights = {}
    if "pickup_weights" in request.form:
        weights = json.loads(request.form["pickup_weights"])

    # Preparo resultados individuales y total
    results = []
    total = 0.0

    for m in maniobras:
        # arma el dict de entrada con llaves EXACTAS de FEATURES
        inp = {
            "rig_name": rig,
            "loc_fed_lease_no": lease,
            "field_name": field,
            "Eventos Normalizados": evento_key,
            "MANIOBRAS NORMALIZADAS": m,
            "pickup_weight": float(weights.get(m, 0)),
            "GEO_LATITUDE": lat,
            "GEO_LONGITUDE": lon
        }
        dur = predict_single(inp)
        results.append({"maniobra": m, "duracion": round(dur, 2)})
        total += dur

    return render_template("results.html", results=results, total=round(total, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

