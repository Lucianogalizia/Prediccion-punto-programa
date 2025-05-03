from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Cargamos modelos y mappings (con espacios en las claves) ---
model1   = joblib.load("models/model1.pkl")
mapping1 = joblib.load("models/mapping1.pkl")
model2   = joblib.load("models/model2.pkl")
mapping2 = joblib.load("models/mapping2.pkl")

# --- Definimos FEATURES usando los mismos nombres con espacios ---
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

# --- Target transformer ---
log_tr = FunctionTransformer(np.log1p, inverse_func=np.expm1)

# --- Carga de coordenadas de pozos ---
coords_df = pd.read_excel("data/coordenadas1.xlsx", dtype={"POZO": str})
for c in ["GEO_LATITUDE", "GEO_LONGITUDE"]:
    coords_df[c] = (
        coords_df[c]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

# --- Frequency encoding para datos nuevos ---
def apply_fe_new(df_new, mapping):
    df = df_new.copy()
    for col, m in mapping.items():
        df[col] = df[col].map(m).fillna(0)
    return df

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
    return float(log_tr.inverse_transform(pred_t.reshape(-1, 1))[0, 0])

# --- Ruta principal ---
@app.route("/")
def home():
    rig_names   = list(mapping1["rig_name"].index)
    lease_nos   = list(mapping1["loc_fed_lease_no"].index)
    field_names = list(mapping1["field_name"].index)
    eventos     = list(mapping1["Eventos Normalizados"].index)
    maniobras   = sorted(
        set(
            mapping1["MANIOBRAS NORMALIZADAS"].index.tolist() +
            mapping2["MANIOBRAS NORMALIZADAS"].index.tolist()
        )
    )
    pozos       = coords_df["POZO"].tolist()
    maniobras_2 = list(mapping2["MANIOBRAS NORMALIZADAS"].index)

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

# --- Autocompletar lat/lon ---
@app.route("/coords/<pozo>")
def coords(pozo):
    row = coords_df[coords_df["POZO"] == pozo].iloc[0]
    return jsonify(lat=row["GEO_LATITUDE"], lon=row["GEO_LONGITUDE"])

# --- Predicción por lote ---
@app.route("/predict", methods=["POST"])
def do_predict():
    # Campos fijos
    rig    = request.form["rig_name"]
    lease  = request.form["loc_fed_lease_no"]
    field  = request.form["field_name"]
    # viene con guión bajo desde el form
    evento_key = request.form["Eventos_Normalizados"]
    lat    = float(request.form["GEO_LATITUDE"])
    lon    = float(request.form["GEO_LONGITUDE"])

    # Maniobras seleccionadas (múltiple)
    maniobras = request.form.getlist("MANIOBRAS_NORMALIZADAS")

    # JSON oculto con pesos por maniobra
    weights = {}
    if "pickup_weights" in request.form:
        weights = json.loads(request.form["pickup_weights"])

    results = []
    total = 0.0

    for m in maniobras:
        # Armo el dict con las **claves con espacios** que espera el modelo
        inp = {
            "rig_name":               rig,
            "loc_fed_lease_no":       lease,
            "field_name":             field,
            "Eventos Normalizados":   evento_key,
            "MANIOBRAS NORMALIZADAS": m,
            "pickup_weight":          float(weights.get(m, 0)),
            "GEO_LATITUDE":           lat,
            "GEO_LONGITUDE":          lon
        }
        dur = predict_single(inp)
        results.append({"maniobra": m, "duracion": round(dur, 2)})
        total += dur

    return render_template("results.html", results=results, total=round(total, 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


