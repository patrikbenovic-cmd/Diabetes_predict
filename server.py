# ============================================================
#  server.py  —  Univerzálny backend pre ML modely
#  Podporuje: Orange, sklearn, joblib, xgboost, lightgbm,
#             keras/tensorflow, torch, onnx
# ============================================================
import os
import json
import pickle
import tempfile
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    import Orange
except ImportError:
    Orange = None

# Adresár, kde je tento skript (funguje aj na serveri)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Debug endpoint povolený iba lokálne (na serveri vypnutý)
DEBUG_ENDPOINTS = os.environ.get("FLASK_DEBUG_ENDPOINTS", "1") == "1"

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


# ============================================================
#  1) REGISTER MODELOV
# ============================================================
MODELS_CONFIG = [
    {
        "id": "diabetes",
        "name": "Predikcia diabetu",
        "description": "Neurónová sieť (MLP) trénovaná v Orange",
        # ✅ RELATÍVNA CESTA — funguje kdekoľvek
        "path": os.path.join(BASE_DIR, "Diabeter_neural_network.pkcls"),
        "kind": "auto",
        "task": "classification",

        # 1 = NEMÁ, 0 = MÁ (podľa tvojho modelu)
        "class_labels": {0: "Má diabetes", 1: "Nemá diabetes"},
        "positive_label": "Má diabetes",

        "metrics": {
            "Presnosť": "75.0 %",
            "AUC skóre": "0.583",
            "F1-skóre": "0.718",
        },
        "metrics_note": "Random Sampling, 50 opakovaní, 80 % trénovacia množina (Orange).",
        "fields": None,
    },

    # --- Ďalšie modely pridávaj sem ---
]


# ============================================================
#  2) NAČÍTANIE MODELOV
# ============================================================
def detect_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".pkcls": "orange",
        ".pickle": "orange",
        ".pkl": "joblib",
        ".joblib": "joblib",
        ".h5": "keras",
        ".keras": "keras",
        ".pt": "torch",
        ".pth": "torch",
        ".onnx": "onnx",
        ".json": "xgboost",
        ".txt": "xgboost",
        ".model": "xgboost",
    }.get(ext, "joblib")


def load_model(path: str, kind: str):
    if kind == "auto":
        kind = detect_kind(path)

    if kind == "orange":
        if Orange is None:
            raise RuntimeError("Chýba Orange3. Nainštaluj: py -m pip install orange3")
        with open(path, "rb") as f:
            return pickle.load(f), kind

    if kind in ("sklearn", "joblib"):
        import joblib
        try:
            return joblib.load(path), "sklearn"
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f), "sklearn"

    if kind == "xgboost":
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(path)
        return b, "xgboost"

    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=path), "lightgbm"

    if kind == "keras":
        import tensorflow as tf
        return tf.keras.models.load_model(path), "keras"

    if kind == "torch":
        import torch
        return torch.load(path, map_location="cpu"), "torch"

    if kind == "onnx":
        import onnxruntime as ort
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"]), "onnx"

    raise ValueError(f"Nepodporovaný typ: {kind}")


# ============================================================
#  3) AUTO-DETEKCIA VSTUPNÝCH POLÍ
# ============================================================
def fields_from_orange(model):
    grouped, singles = {}, []
    for var in model.domain.attributes:
        name = var.name
        if "=" in name:
            prefix, val = name.split("=", 1)
            grouped.setdefault(prefix, []).append(val)
        else:
            singles.append((name, var))

    fields = []
    for name, var in singles:
        if var.is_discrete:
            fields.append({
                "name": name, "label": name.replace("_", " "),
                "type": "categorical",
                "options": [str(v) for v in var.values],
                "default": str(var.values[0]),
            })
        elif var.is_continuous:
            fields.append({
                "name": name, "label": name.replace("_", " "),
                "type": "number", "step": 0.1, "default": 0.0,
            })
        else:
            fields.append({
                "name": name, "label": name.replace("_", " "),
                "type": "text", "default": "",
            })

    for prefix, vals in grouped.items():
        vals_sorted = sorted(vals, key=lambda v: (len(v), v))
        if set(vals_sorted) == {"0", "1"}:
            options = ["0", "1"]
            labels = {"0": "Nie", "1": "Áno"}
        else:
            options = vals_sorted
            labels = {v: v for v in vals_sorted}
        fields.append({
            "name": prefix, "label": prefix.replace("_", " "),
            "type": "categorical",
            "options": options, "option_labels": labels,
            "default": options[0],
        })
    return fields


def fields_from_xgboost(model):
    try:
        names = model.feature_names
    except Exception:
        names = None
    if not names:
        return None
    return [{"name": n, "label": n.replace("_", " "),
             "type": "number", "step": 0.1, "default": 0.0} for n in names]


def fields_from_sklearn(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    return [{"name": str(n), "label": str(n).replace("_", " "),
             "type": "number", "step": 0.1, "default": 0.0} for n in names]


def autodetect_fields(model, kind):
    try:
        if kind == "orange":
            return fields_from_orange(model)
        if kind == "xgboost":
            return fields_from_xgboost(model)
        if kind == "sklearn":
            return fields_from_sklearn(model)
    except Exception:
        traceback.print_exc()
    return None


# ============================================================
#  4) PREDIKCIA
# ============================================================
def predict_orange(model, data: dict):
    attrs = [v.name for v in model.domain.attributes]
    row = {}
    for a in attrs:
        if a in data:
            row[a] = data[a]
        elif "=" in a:
            prefix, val = a.split("=", 1)
            if prefix in data:
                cur = str(data[prefix])
                try:
                    match = (cur == val) or (float(cur) == float(val))
                except (ValueError, TypeError):
                    match = (cur == val)
                row[a] = 1.0 if match else 0.0
            else:
                row[a] = 0.0
        else:
            row[a] = 0.0

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.close()
    try:
        pd.DataFrame([row]).to_csv(tmp.name, index=False)
        t_raw = Orange.data.Table(tmp.name)
        t = Orange.data.Table.from_table(model.domain, t_raw)
    finally:
        os.remove(tmp.name)

    pred = model(t)
    pred_idx = int(pred[0])
    probs = model(t, Orange.classification.Model.Probs)[0]
    classes = [str(v) for v in model.domain.class_var.values]
    return pred_idx, classes, list(map(float, probs))


def _build_dataframe(fields, data):
    return pd.DataFrame([{f["name"]: data[f["name"]] for f in fields}])


def predict_sklearn(model, data, fields):
    X = _build_dataframe(fields, data)
    pred_idx = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        probs = np.array([0.0, 1.0]) if pred_idx == 1 else np.array([1.0, 0.0])
    classes = [str(c) for c in getattr(model, "classes_", range(len(probs)))]
    return pred_idx, classes, list(map(float, probs))


def predict_xgboost(model, data, fields):
    import xgboost as xgb
    X = _build_dataframe(fields, data)
    dm = xgb.DMatrix(X)
    raw = model.predict(dm)
    if raw.ndim == 1:
        p1 = float(raw[0]); p0 = 1.0 - p1
        idx = int(p1 >= 0.5)
        return idx, ["0", "1"], [p0, p1]
    row = raw[0]
    idx = int(np.argmax(row))
    return idx, [str(i) for i in range(len(row))], list(map(float, row))


def predict_lightgbm(model, data, fields):
    X = _build_dataframe(fields, data)
    raw = model.predict(X)
    if raw.ndim == 1:
        p1 = float(raw[0]); p0 = 1.0 - p1
        idx = int(p1 >= 0.5)
        return idx, ["0", "1"], [p0, p1]
    row = raw[0]
    idx = int(np.argmax(row))
    return idx, [str(i) for i in range(len(row))], list(map(float, row))


def predict_keras(model, data, fields):
    X = _build_dataframe(fields, data).values.astype("float32")
    raw = model.predict(X, verbose=0)[0]
    idx = int(np.argmax(raw))
    return idx, [str(i) for i in range(len(raw))], list(map(float, raw))


def predict_torch(model, data, fields):
    import torch
    X = _build_dataframe(fields, data).values.astype("float32")
    t = torch.from_numpy(X)
    model.eval()
    with torch.no_grad():
        out = model(t)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        probs = torch.softmax(out, dim=1)[0].numpy()
    idx = int(np.argmax(probs))
    return idx, [str(i) for i in range(len(probs))], list(map(float, probs))


def predict_onnx(session, data, fields):
    X = _build_dataframe(fields, data).values.astype("float32")
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: X})[0][0]
    if len(out.shape) == 0 or out.size == 1:
        p1 = float(out.ravel()[0]); p0 = 1.0 - p1
        idx = int(p1 >= 0.5)
        return idx, ["0", "1"], [p0, p1]
    idx = int(np.argmax(out))
    return idx, [str(i) for i in range(len(out))], list(map(float, out))


PREDICTORS = {
    "orange":    predict_orange,
    "sklearn":   predict_sklearn,
    "xgboost":   predict_xgboost,
    "lightgbm":  predict_lightgbm,
    "keras":     predict_keras,
    "torch":     predict_torch,
    "onnx":      predict_onnx,
}


# ============================================================
#  5) INICIALIZÁCIA REGISTRA
# ============================================================
REGISTRY = {}


def init_registry():
    for cfg in MODELS_CONFIG:
        try:
            model, kind = load_model(cfg["path"], cfg.get("kind", "auto"))
            fields = cfg.get("fields") or autodetect_fields(model, kind)
            if not fields:
                print(f"[!] {cfg['id']}: nepodarilo sa zistiť polia.")
                fields = []
            REGISTRY[cfg["id"]] = {
                "cfg": cfg, "model": model, "kind": kind, "fields": fields,
            }
            print(f"[✓] Načítaný model '{cfg['id']}' ({kind}), polí: {len(fields)}")
        except Exception as e:
            print(f"[✗] Nepodarilo sa načítať '{cfg['id']}': {e}")
            traceback.print_exc()


# ============================================================
#  6) API
# ============================================================
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    out = []
    for mid, item in REGISTRY.items():
        cfg = item["cfg"]
        out.append({
            "id": mid,
            "name": cfg["name"],
            "description": cfg.get("description", ""),
            "kind": item["kind"],
            "task": cfg.get("task", "classification"),
            "metrics": cfg.get("metrics", {}),
            "metrics_note": cfg.get("metrics_note", ""),
            "fields": item["fields"],
        })
    return jsonify({"models": out})


@app.route("/api/predict/<model_id>", methods=["POST"])
def predict(model_id):
    if model_id not in REGISTRY:
        return jsonify({"error": f"Neznámy model: {model_id}"}), 404

    item = REGISTRY[model_id]
    cfg = item["cfg"]
    model = item["model"]
    kind = item["kind"]
    fields = item["fields"]

    try:
        raw = request.get_json(force=True)

        # --- Pretypovanie vstupov ---
        data = {}
        for f in fields:
            v = raw.get(f["name"])
            if v is None or v == "":
                v = f.get("default", 0)
            if f["type"] == "number":
                data[f["name"]] = float(v)
            else:
                data[f["name"]] = str(v)

        # --- Predikcia ---
        if kind == "orange":
            pred_idx, classes, probs = predict_orange(model, data)
        else:
            pred_idx, classes, probs = PREDICTORS[kind](model, data, fields)

        # --- Mapovanie index -> label ---
        labels_map = cfg.get("class_labels") or {}
        if labels_map:
            label = labels_map.get(
                pred_idx,
                classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
            )
        else:
            label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

        # --- Pozitivita podľa labelu ---
        positive_label = cfg.get("positive_label")
        if positive_label:
            is_positive = (label == positive_label)
        else:
            is_positive = (pred_idx != 0)

        # --- Rozloženie pravdepodobností ---
        prob_list = []
        for i, p in enumerate(probs):
            if labels_map:
                lbl = labels_map.get(i, classes[i] if i < len(classes) else str(i))
            else:
                lbl = classes[i] if i < len(classes) else str(i)
            prob_list.append({"label": lbl, "probability": float(p)})

        return jsonify({
            "result": label,
            "predicted_index": pred_idx,
            "is_positive": is_positive,
            "confidence": float(max(probs)) if probs else 1.0,
            "probabilities": prob_list,
            "task": cfg.get("task", "classification"),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/<model_id>", methods=["GET"])
def debug(model_id):
    """Pomocný endpoint – zobrazí poradie tried v modeli (len lokálne)."""
    if not DEBUG_ENDPOINTS:
        return jsonify({"error": "Debug endpoint je vypnutý"}), 403
    if model_id not in REGISTRY:
        return jsonify({"error": "Neznámy model"}), 404
    item = REGISTRY[model_id]
    m = item["model"]
    out = {"kind": item["kind"]}
    try:
        if item["kind"] == "orange":
            out["class_values"] = [str(v) for v in m.domain.class_var.values]
        elif hasattr(m, "classes_"):
            out["class_values"] = [str(c) for c in m.classes_]
    except Exception as e:
        out["error"] = str(e)
    return jsonify(out)


# ============================================================
#  7) ŠTART
# ============================================================
# Zavolá sa VŽDY — aj pod gunicornom, aj lokálne
init_registry()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
