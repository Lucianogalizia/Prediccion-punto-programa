from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np              
import joblib
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

app = Flask(__name__, static_folder="static", template_folder="templates")

# — Cargamos modelos/mappings al iniciar —
model1, mapping1 = joblib.load("models/model1.pkl"), joblib.load("models/mapping1.pkl")
model2, mapping2 = joblib.load("models/model2.pkl"), joblib.load("models/mapping2.pkl")

# --- Feature lists ---
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


log_tr = FunctionTransformer(np.log1p, inverse_func=np.expm1)

# — Cargamos coordenadas —
coords_df = pd.read_excel("data/coordenadas1.xlsx", dtype={"POZO": str})
for c in ["GEO_LATITUDE","GEO_LONGITUDE"]:
    coords_df[c] = coords_df[c].astype(str).str.replace(",",".").astype(float)

# — Helpers de frequency encoding/inferencia (tal cual en notebook) —
def apply_fe_new(X_new,mapping):
    X=X_new.copy()
    for col,m in mapping.items():
        X[col]=X[col].map(m).fillna(0)
    return X

def predict(dict_in):
    mani = dict_in["MANIOBRAS_NORMALIZADAS"]
    if mani in mapping2["MANIOBRAS NORMALIZADAS"].index:
        mdl, mp, feats = model2, mapping2, FEATURES2
    else:
        mdl, mp, feats = model1, mapping1, FEATURES1
    df = pd.DataFrame([dict_in])[feats]
    df_enc = apply_fe_new(df, mp)
    pred_t = mdl.predict(xgb.DMatrix(df_enc))
    return float(log_tr.inverse_transform(pred_t.reshape(-1,1))[0,0])

# — Rutas —
@app.route("/")
def home():
    return render_template("form.html",
        rig_names=list(mapping1["rig_name"].index),
        lease_nos=list(mapping1["loc_fed_lease_no"].index),
        field_names=list(mapping1["field_name"].index),
        eventos=list(mapping1["Eventos Normalizados"].index),
        maniobras=sorted(set(mapping1["MANIOBRAS NORMALIZADAS"].index.tolist()+
                             mapping2["MANIOBRAS NORMALIZADAS"].index.tolist())),
        pozos=coords_df["POZO"].tolist(),
        # <-- AÑADE ESTA LÍNEA -->
        maniobras_2=list(mapping2["MANIOBRAS NORMALIZADAS"].index)                   
    )

@app.route("/coords/<pozo>")
def coords(pozo):
    row = coords_df[coords_df["POZO"]==pozo].iloc[0]
    return jsonify(lat=row["GEO_LATITUDE"], lon=row["GEO_LONGITUDE"])

@app.route("/predict", methods=["POST"])
def do_predict():
    data = request.form.to_dict()
    # parseos numéricos
    data["GEO_LATITUDE"], data["GEO_LONGITUDE"] = float(data["GEO_LATITUDE"]), float(data["GEO_LONGITUDE"])
    data["pickup_weight"] = float(data.get("pickup_weight","0") or 0)
    duration = predict(data)
    return render_template("results.html", duration=round(duration,2))

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080)
