#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buscador.py — Búsqueda rápida de hiperparámetros (RF y XGBoost) con comparador final.

Novedades:
- Evalúa cada intento en TRAIN, VAL y TEST.
- Rankings finales:
  1) Top-3 por validación→test (rank por AUC val; se muestra su test).
  2) Top-3 por test (rank por AUC test).
  3) Top-3 mejor generalización train-test (menor |AUC_train − AUC_test|, desempata por AUC_test).
"""

import os, sys, math, json, datetime, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, ParameterSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# XGBoost (opcional)
try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

# ======================
# Rutas y constantes
# ======================
SEED = 1234

THIS_FILE = Path(__file__).resolve()
LIB_DIR = THIS_FILE.parent
REPO_ROOT = LIB_DIR.parent

DEFAULT_DATA = REPO_ROOT / "Dataset" / "Procesados" / "descriptores.csv"

# Columnas meta que no deben usarse como features (coincide con model.py)
META_FORBIDDEN = {
    "Standard Value","Molecule ChEMBL ID","Smiles","canonical_smiles",
    "Standard Type","Standard Relation","Standard Units","Target ChEMBL ID","activo",
    "PAINS_flag","PAINS_terms","Chelator_flag","Chelator_terms",
    "Lipinski_violations","Veber_violations",
}

# ======================
# Utilidades
# ======================
def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def set_seed(seed: int = SEED):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")

def build_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preproc = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols)],
        remainder="drop",
        sparse_threshold=0.0
    )
    return preproc

def stratified_train_val_test(X: pd.DataFrame, y: pd.Series,
                              train_ratio: float, val_ratio: float,
                              seed: int = SEED):
    assert 0 < train_ratio < 1, "train debe estar en (0,1)"
    assert 0 <= val_ratio < 1, "val debe estar en [0,1)"
    assert train_ratio + val_ratio < 1, "train+val debe ser < 1"
    test_ratio = 1.0 - train_ratio - val_ratio

    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    (train_idx, rest_idx) = next(sss1.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_rest,  y_rest  = X.iloc[rest_idx],  y.iloc[rest_idx]

    val_share = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_share, random_state=seed)
    (val_idx_rel, test_idx_rel) = next(sss2.split(X_rest, y_rest))
    X_val, y_val   = X_rest.iloc[val_idx_rel],  y_rest.iloc[val_idx_rel]
    X_test, y_test = X_rest.iloc[test_idx_rel], y_rest.iloc[test_idx_rel]
    return X_train, y_train, X_val, y_val, X_test, y_test

def evaluate_all(model, preproc, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> Dict[str, Any]:
    Xp = preproc.transform(X)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(Xp)[:,1]
    elif hasattr(model, "decision_function"):
        z = model.decision_function(Xp)
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        prob = z
    else:
        prob = model.predict(Xp).astype(float)

    pred = (prob >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc":  average_precision_score(y, prob),
        "f1":      f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall":  recall_score(y, pred, zero_division=0),
        "cm":      confusion_matrix(y, pred),
    }
    return out

def build_label_by_fraction(values: pd.Series, fraction: float, seed: int = SEED) -> Tuple[pd.Series, float]:
    """
    Marca como 1 el fraction% de filas con menor IC50 (Standard Value).
    Si hay empates en el valor de corte, completa al azar (semilla fija)
    hasta cumplir exactamente la fracción.
    Devuelve (y, corte_nm) donde corte_nm es el IC50 máximo dentro de los seleccionados=1.
    """
    vals = pd.to_numeric(values, errors="coerce")
    valid = vals.notna()
    idx = np.arange(len(vals))
    idx_valid = idx[valid.values]
    vals_valid = vals[valid.values].values

    order = np.argsort(vals_valid, kind="mergesort")        # estable
    idx_sorted = idx_valid[order]
    vals_sorted = vals_valid[order]

    k = int(math.floor(fraction * len(vals_sorted)))
    k = max(1, min(k, len(vals_sorted)))

    cut_val_nominal = vals_sorted[k-1]
    mask_lt = vals_sorted < cut_val_nominal
    idx_lt = idx_sorted[mask_lt]
    count_lt = len(idx_lt)

    mask_eq = vals_sorted == cut_val_nominal
    idx_eq = idx_sorted[mask_eq]

    rng = np.random.RandomState(seed)
    remaining = k - count_lt
    if remaining <= 0:
        chosen_eq = np.array([], dtype=idx_eq.dtype)
    else:
        chosen_eq = idx_eq if remaining >= len(idx_eq) else rng.choice(idx_eq, size=remaining, replace=False)

    chosen = set(idx_lt.tolist() + chosen_eq.tolist())
    y = pd.Series(0, index=range(len(values)), dtype=int)
    y.loc[list(chosen)] = 1

    corte_nm = float(np.nanmax(vals.loc[y.astype(bool)])) if y.any() else float(cut_val_nominal)
    return y, corte_nm

# ======================
# Búsquedas
# ======================
def search_rf(X_train, y_train, X_val, y_val, numeric_cols: List[str],
              trials: int = 5, seed: int = SEED):
    space = {
        "n_estimators": [300, 500, 800, 1200, 1600, 1800, 2000, 2200],
        "max_depth":    [None, 10, 20, 30, 40, 50],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6],
        "max_features": ["sqrt", "log2", None],
    }
    rng = np.random.RandomState(seed)
    samples = list(ParameterSampler(space, n_iter=trials, random_state=rng))

    board = []
    print(f"[RF] Probando {len(samples)} configuraciones...")
    for i, cfg in enumerate(samples, start=1):
        pre = build_preprocessor(numeric_cols)
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg["max_features"],
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        X_trp = pre.fit_transform(X_train); X_vp = pre.transform(X_val)
        model.fit(X_trp, y_train)

        # métricas train/val
        m_tr = evaluate_all(model, pre, X_train, y_train)
        m_v  = evaluate_all(model, pre, X_val,   y_val)

        board.append((
            m_v["roc_auc"], m_v["f1"], cfg, m_tr, m_v, model, pre, "rf"
        ))
        print(f"[RF {i:02d}] cfg={cfg} | TRAIN AUC={m_tr['roc_auc']:.4f} F1={m_tr['f1']:.4f}  ||  VAL AUC={m_v['roc_auc']:.4f} F1={m_v['f1']:.4f}")

    board.sort(key=lambda x: (x[0], x[1]), reverse=True)  # por AUC val, luego F1
    print("\n[RF] TOP por validación:")
    for i, row in enumerate(board[:min(5, len(board))], start=1):
        aucv, f1v, cfg, mtr, mv, _, _, _ = row
        print(f"  {i:02d}) VAL AUC={aucv:.4f} F1={f1v:.4f} | cfg={cfg}")
    return board

def search_xgb(X_train, y_train, X_val, y_val, numeric_cols: List[str],
               trials: int = 5, seed: int = SEED):
    if not _HAVE_XGB:
        print("[XGB] xgboost no está instalado (pip install xgboost). Se omite.")
        return []

    space = {
        "n_estimators": [400, 800, 1200, 1600, 1800, 2000],
        "max_depth":    [4, 5, 6, 7, 8],
        "learning_rate": [0.05, 0.1, 0.15, 0.2],
        "subsample":    [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda":   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    }
    rng = np.random.RandomState(seed)
    samples = list(ParameterSampler(space, n_iter=trials, random_state=rng))

    board = []
    print(f"[XGB] Probando {len(samples)} configuraciones...")
    for i, cfg in enumerate(samples, start=1):
        pre = build_preprocessor(numeric_cols)
        model = xgb.XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_lambda=cfg["reg_lambda"],
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="auc",
            verbosity=0,
        )
        X_trp = pre.fit_transform(X_train); X_vp = pre.transform(X_val)
        model.fit(X_trp, y_train)

        m_tr = evaluate_all(model, pre, X_train, y_train)
        m_v  = evaluate_all(model, pre, X_val,   y_val)

        board.append((
            m_v["roc_auc"], m_v["f1"], cfg, m_tr, m_v, model, pre, "xgb"
        ))
        print(f"[XGB {i:02d}] cfg={cfg} | TRAIN AUC={m_tr['roc_auc']:.4f} F1={m_tr['f1']:.4f}  ||  VAL AUC={m_v['roc_auc']:.4f} F1={m_v['f1']:.4f}")

    board.sort(key=lambda x: (x[0], x[1]), reverse=True)
    print("\n[XGB] TOP por validación:")
    for i, row in enumerate(board[:min(5, len(board))], start=1):
        aucv, f1v, cfg, mtr, mv, _, _, _ = row
        print(f"  {i:02d}) VAL AUC={aucv:.4f} F1={f1v:.4f} | cfg={cfg}")
    return board

# ======================
# CLI
# ======================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Búsqueda rápida RF/XGB para el TFM (con comparador final).")
    p.add_argument("--dir", type=str, default=str(DEFAULT_DATA), help="Ruta a descriptores.csv")
    p.add_argument("--ic50", type=float, default=0.5, help="Fracción de activos (0<f<=1). Por defecto 0.5.")
    p.add_argument("--train", type=float, default=0.75, help="Proporción de entrenamiento. Defecto 0.75.")
    p.add_argument("--val", type=float, default=0.15, help="Proporción de validación. Defecto 0.15.")
    p.add_argument("--trials", type=int, default=5, help="Nº de configuraciones a probar por algoritmo.")
    p.add_argument("--algo", type=str, choices=["both","rf","xgb"], default="both", help="Qué algoritmo(s) evaluar.")
    p.add_argument("--seed", type=int, default=SEED, help="Semilla para reproducibilidad.")
    return p.parse_args()

# ======================
# Main
# ======================
def main():
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {data_path}")

    print(f"[INFO] Cargando CSV: {data_path}")
    df = read_csv_robust(data_path)
    print(f"[INFO] CSV cargado. Filas: {len(df)}, Columnas: {len(df.columns)}")

    if "Standard Value" not in df.columns:
        raise ValueError("Falta la columna 'Standard Value' para etiquetar por IC50.")

    # Etiquetado por fracción + umbral nM
    frac = float(args.ic50)
    if not (0.0 < frac <= 1.0):
        raise ValueError("--ic50 debe estar en (0,1].")
    df = df.copy()
    df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
    df = df.dropna(subset=["Standard Value"]).reset_index(drop=True)
    y, corte_nm = build_label_by_fraction(df["Standard Value"], fraction=frac, seed=args.seed)
    df["activo"] = y.astype(int)

    vc = df["activo"].value_counts().sort_index()
    print(f"[INFO] Distribución de 'activo' (0/1): {dict(vc)}  (ic50={frac})")
    print(f"[INFO] Umbral IC50 usado (nM): {corte_nm:.2f}")

    # Selección de features numéricas
    numeric_cols = [c for c in df.columns if c not in META_FORBIDDEN and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No hay columnas numéricas de descriptores disponibles.")
    X = df[numeric_cols].copy()
    y = df["activo"].copy()

    # Splits
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_train_val_test(
        X, y, float(args.train), float(args.val), args.seed
    )

    # Mostrar un pequeño head de entrenamiento
    print("[INFO] Head de X_train (primeras 5 filas, columnas truncadas):")
    with pd.option_context('display.max_columns', 12):
        print(X_train.head(5))

    # Búsquedas
    boards = []
    if args.algo in ("both","rf"):
        rf_board = search_rf(X_train, y_train, X_val, y_val, numeric_cols, trials=int(args.trials), seed=args.seed)
        boards.extend(rf_board)
    if args.algo in ("both","xgb"):
        xgb_board = search_xgb(X_train, y_train, X_val, y_val, numeric_cols, trials=int(args.trials), seed=args.seed)
        boards.extend(xgb_board)

    if not boards:
        print("\n[AVISO] No se han evaluado modelos (¿sin features numéricas o sin xgboost?)")
        print("[FIN] Búsqueda completada.")
        return 0

    # ===== Comparador final: evaluar TEST para TODOS los intentos y rankear =====
    records = []
    for (val_auc, val_f1, cfg, m_tr, m_v, model, pre, algo) in boards:
        m_te = evaluate_all(model, pre, X_test, y_test)
        rec = {
            "algo": algo,
            "cfg": cfg,
            "train": m_tr,
            "val": m_v,
            "test": m_te,
        }
        records.append(rec)

    def print_entry(idx: int, rec: Dict[str, Any], tag: str):
        cfg = rec["cfg"]; algo = rec["algo"]
        mt, mv, mte = rec["train"], rec["val"], rec["test"]
        print(f"  {idx:02d}) [{algo.upper()}] {tag}")
        print(f"      cfg={json.dumps(cfg)}")
        print(f"      TRAIN: AUC={mt['roc_auc']:.4f} F1={mt['f1']:.4f}")
        print(f"      VAL  : AUC={mv['roc_auc']:.4f} F1={mv['f1']:.4f}")
        print(f"      TEST : AUC={mte['roc_auc']:.4f} F1={mte['f1']:.4f}")

    # Top-3 por validación (AUC val, luego F1 val)
    top_by_val = sorted(records, key=lambda r: (r["val"]["roc_auc"], r["val"]["f1"]), reverse=True)[:3]
    print("\n[COMPARADOR] Top-3 por validación → test (rank por AUC val)")
    for i, rec in enumerate(top_by_val, start=1):
        print_entry(i, rec, "mejores por VALIDACIÓN")

    # Top-3 por test (AUC test, luego F1 test)
    top_by_test = sorted(records, key=lambda r: (r["test"]["roc_auc"], r["test"]["f1"]), reverse=True)[:3]
    print("\n[COMPARADOR] Top-3 por TEST (rank por AUC test)")
    for i, rec in enumerate(top_by_test, start=1):
        print_entry(i, rec, "mejores por TEST")

    # Top-3 mejor generalización train-test (gap pequeño, desempate por AUC test)
    top_by_gap = sorted(
        records,
        key=lambda r: (abs(r["train"]["roc_auc"] - r["test"]["roc_auc"]), -r["test"]["roc_auc"])
    )[:3]
    print("\n[COMPARADOR] Top-3 mejor generalización train-test (menor gap |AUC_train − AUC_test|)")
    for i, rec in enumerate(top_by_gap, start=1):
        gap = abs(rec["train"]["roc_auc"] - rec["test"]["roc_auc"])
        print_entry(i, rec, f"gap={gap:.4f}")

    print("\n[FIN] Búsqueda y comparación completadas.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
