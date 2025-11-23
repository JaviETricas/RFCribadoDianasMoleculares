#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predictor.py — Predice actividad (0/1) sobre Dataset/Prediccion/descriptores.csv
usando un modelo guardado en Models/<nombre>/checkpoints/. Si existen columnas de
filtros (PAINS, queladores, Lipinski/Veber), las añade como una columna textual
('filtros') al final. **Solo genera un CSV resumido** con:
[id, prob_activo, activo_pred, filtros]

Ejemplos:
  python predictor.py                          # usa Models/base y Dataset/Prediccion/descriptores.csv
  python predictor.py --model base             # modelo 'base'
  python predictor.py --dir "C:\\ruta\\file.csv"  # otro CSV
  python predictor.py --threshold 0.6
"""

import sys, datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib

# ======================
# Rutas relativas
# ======================
THIS_FILE = Path(__file__).resolve()
LIB_DIR   = THIS_FILE.parent
REPO_ROOT = LIB_DIR.parent

DATASET_DIR    = REPO_ROOT / "Dataset"
PRED_DIR       = DATASET_DIR / "Prediccion"
DEFAULT_CSV    = PRED_DIR / "descriptores.csv"

MODELS_DIR     = REPO_ROOT / "Models"
RESULTS_DIR    = REPO_ROOT / "Resultados"

# ======================
# Config / Constantes
# ======================
SEED = 1234
DEFAULT_MODEL_NAME = "base"
DEFAULT_THRESHOLD  = 0.5

# Columnas meta a no usar como features
META_FORBIDDEN = {
    "Standard Value","Molecule ChEMBL ID","Smiles","canonical_smiles",
    "Standard Type","Standard Relation","Standard Units","Target ChEMBL ID","activo",
    # filtros (si existen, se excluyen de features):
    "PAINS_flag","PAINS_terms","Chelator_flag","Chelator_terms",
    "Lipinski_violations","Veber_violations",
}

FILTER_COLS = [
    "PAINS_flag","PAINS_terms","Chelator_flag","Chelator_terms",
    "Lipinski_violations","Veber_violations"
]

# ======================
# Utilidades
# ======================
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")

def pick_id_column(df: pd.DataFrame) -> str:
    for c in ["Molecule ChEMBL ID","canonical_smiles","Smiles"]:
        if c in df.columns:
            return c
    return "__index__"

def get_required_feature_names(preproc) -> List[str]:
    # Nuestro preproc es un ColumnTransformer con el primer transformer 'num'
    try:
        for name, trans, cols in preproc.transformers_:
            if name == "num":
                return list(cols)
    except Exception:
        pass
    return []

def truthy(x) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    s = str(x).strip().lower()
    if s in {"1","true","yes","si","sí","y"}:
        return True
    try:
        return float(s) != 0.0
    except Exception:
        return bool(s)

def build_filters_summary(row: pd.Series) -> str:
    msgs = []
    # PAINS
    if "PAINS_flag" in row and truthy(row["PAINS_flag"]):
        term = row.get("PAINS_terms", "")
        if isinstance(term, float) and np.isnan(term):
            term = ""
        msgs.append(f"PAINS: {term if term else 'sí'}")
    # Queladores
    if "Chelator_flag" in row and truthy(row["Chelator_flag"]):
        term = row.get("Chelator_terms", "")
        if isinstance(term, float) and np.isnan(term):
            term = ""
        msgs.append(f"Quelación: {term if term else 'sí'}")
    # Lipinski/Veber
    lv = row.get("Lipinski_violations", np.nan)
    vv = row.get("Veber_violations",    np.nan)
    try:
        if pd.notna(lv) and int(lv) > 0:
            msgs.append(f"Lipinski violaciones: {int(lv)}")
    except Exception:
        pass
    try:
        if pd.notna(vv) and int(vv) > 0:
            msgs.append(f"Veber violaciones: {int(vv)}")
    except Exception:
        pass
    return " | ".join(msgs)

# ======================
# CLI
# ======================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Predicción de actividad (0/1) con modelo entrenado.")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                   help="Nombre del modelo en Models/ (por defecto 'base').")
    p.add_argument("--dir", type=str, default=str(DEFAULT_CSV),
                   help="CSV de descriptores para predecir (por defecto Dataset/Prediccion/descriptores.csv).")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="Umbral para activo_pred (por defecto 0.5).")
    return p.parse_args()

# ======================
# Principal
# ======================
def main():
    args = parse_args()
    csv_path = Path(args.dir).resolve()

    if not csv_path.exists():
        print(f"[ERROR] No existe el CSV de entrada: {csv_path}")
        return 1

    model_dir = MODELS_DIR / args.model
    ckpt_dir  = model_dir / "checkpoints"
    model_pkl = ckpt_dir / "model.pkl"
    pre_pkl   = ckpt_dir / "preproc.pkl"

    if not model_pkl.exists() or not pre_pkl.exists():
        print(f"[ERROR] No se encontró checkpoint del modelo '{args.model}'. Esperado en:\n  {model_pkl}\n  {pre_pkl}")
        return 1

    print(f"[INFO] Cargando modelo: {model_pkl}")
    model = joblib.load(model_pkl)
    preproc = joblib.load(pre_pkl)

    # Cargar datos
    print(f"[INFO] Cargando CSV de predicción: {csv_path}")
    df = read_csv_robust(csv_path)
    print(f"[INFO] CSV cargado. Filas: {len(df)}, Columnas: {len(df.columns)}")

    # Selección de features: usar exactamente las que requiere el preprocesador
    req_cols = get_required_feature_names(preproc)
    if not req_cols:
        print("[ERROR] No pude recuperar la lista de columnas de features del preprocesador.")
        return 1

    # Faltantes -> NaN (imputación cubrirá)
    missing = [c for c in req_cols if c not in df.columns]
    for c in missing:
        df[c] = np.nan
    if missing:
        print(f"[AVISO] Faltaban {len(missing)} columnas de features; se añadieron como NaN.")

    # Ordenar columnas según req_cols
    X = df[req_cols].copy()

    # Predicción
    Xp = preproc.transform(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(Xp)[:, 1]
    elif hasattr(model, "decision_function"):
        z = model.decision_function(Xp)
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        y_prob = z
    else:
        y_prob = model.predict(Xp).astype(float)

    thr = float(args.threshold)
    y_pred = (y_prob >= thr).astype(int)

    # --- Construir salida RESUMIDA ---
    id_col = pick_id_column(df)
    resumen = pd.DataFrame({
        "id": df[id_col] if id_col in df.columns else df.index,
        "prob_activo": y_prob,
        "activo_pred": y_pred,
    })

    # Añadir columna textual 'filtros' si existen columnas de filtros
    has_any_filters = any(c in df.columns for c in FILTER_COLS)
    if has_any_filters:
        resumen["filtros"] = [build_filters_summary(row) for _, row in df.iterrows()]
    else:
        resumen["filtros"] = ""  # columna vacía si no hay filtros

    # Mostrar en pantalla — resumen
    print("[INFO] Predicciones (resumen: id, prob_activo, activo_pred, filtros):")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(resumen)

    # Guardar **solo un CSV** (resumen)
    ensure_dir(RESULTS_DIR)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = RESULTS_DIR / f"predicciones_{args.model}_{ts}_resumen.csv"
    resumen.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Resumen guardado en: {out_csv}")

    # Resumen corto por consola
    n = len(resumen)
    n1 = int((resumen['activo_pred'] == 1).sum())
    n0 = n - n1
    print(f"[RESUMEN] n={n} | activos_pred=1: {n1} | inactivos_pred=0: {n0} | umbral={thr}")
    if has_any_filters:
        flagged = resumen["filtros"].fillna("").str.len() > 0
        print(f"[RESUMEN] Con alguna alerta de filtros: {int(flagged.sum())} de {n}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
