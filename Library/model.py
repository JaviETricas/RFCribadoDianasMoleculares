#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py — Entrenamiento de modelos para TFM (clasificación de alta actividad) con rutas relativas,
etiquetado por fracción de IC50, prevención de suspensión en Windows, guardado reproducible,
mensajes de progreso, vista de 'activo' y parada temprana (early stopping).

Novedades vs versión anterior:
- Muestra distribución y ejemplo de la columna 'activo' (0/1) tras el etiquetado.
- Muestra head de ENTRENAMIENTO incluyendo 'activo' (no solo features).
- Nuevo flag: --retrain para forzar entrenamiento aunque exista checkpoint.
- Progreso de entrenamiento:
    * RF: incremental con warm_start; imprime AUC/F1 de train y val por bloque; early stopping por paciencia.
    * XGB: early_stopping_rounds y verbose opcional; imprime mejor iteración.
- Guarda SIEMPRE el mejor modelo según validación.
"""

import os, sys, json, math, ctypes, hashlib, datetime, warnings, copy
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

import joblib
import matplotlib
matplotlib.use("Agg")  # para entornos sin display
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import StratifiedShuffleSplit, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state

# Silenciar FutureWarnings ruidosos (p.ej., SHAP)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import opcional de XGBoost y SHAP
try:
    import xgboost as xgb
    _HAVE_XGB = True
except Exception:
    _HAVE_XGB = False

try:
    import shap
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

# ======================
# Constantes y rutas
# ======================
SEED = 1234
DEFAULT_IC50_FRAC = 0.5
DEFAULT_TRAIN = 0.75
DEFAULT_VAL = 0.15
DEFAULT_BASE_MODEL_NAME = "base"
MODE_CHOICES = {"base", "new", "test"}

THIS_FILE = Path(__file__).resolve()
LIB_DIR = THIS_FILE.parent
REPO_ROOT = LIB_DIR.parent

DEFAULT_DATA = REPO_ROOT / "Dataset" / "Procesados" / "descriptores.csv"
MODELS_DIR = REPO_ROOT / "Models"

# Columnas meta que no deben usarse como features
META_FORBIDDEN = {
    "Standard Value","Molecule ChEMBL ID","Smiles","canonical_smiles",
    "Standard Type","Standard Relation","Standard Units","Target ChEMBL ID","activo",
    # columnas nuevas de filtros:
    "PAINS_flag","PAINS_terms","Chelator_flag","Chelator_terms",
    "Lipinski_violations","Veber_violations",
}


# ======================
# Utilidades
# ======================
def prevent_sleep_start():
    """Evita que Windows entre en suspensión durante procesos largos."""
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)

def prevent_sleep_stop():
    if os.name == "nt":
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

def set_global_seed(seed: int = SEED):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def figure_path(dirpath: Path, name: str) -> Path:
    ensure_dir(dirpath)
    return dirpath / name

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

# ======================
# Etiquetado por fracción
# ======================
def build_label_by_fraction(values: pd.Series, fraction: float, seed: int = SEED) -> pd.Series:
    """
    Marca como 1 (activo) el fraction% de filas con menor IC50 (Standard Value).
    Si hay empates en el valor de corte, selecciona aleatoriamente (con semilla) dentro de los empatados
    hasta cumplir exactamente el porcentaje deseado.
    """
    vals = pd.to_numeric(values, errors="coerce")
    valid = vals.notna()
    idx = np.arange(len(vals))
    idx_valid = idx[valid.values]
    vals_valid = vals[valid.values].values

    # Orden ascendente (menor IC50 primero)
    order = np.argsort(vals_valid, kind="mergesort")
    idx_sorted = idx_valid[order]
    vals_sorted = vals_valid[order]

    k = int(math.floor(fraction * len(vals_sorted)))
    k = max(1, min(k, len(vals_sorted)))  # al menos 1 y no más que el total

    # Valor de corte
    cut_val = vals_sorted[k-1]

    # Todo lo menor que el corte entra seguro
    mask_lt = vals_sorted < cut_val
    idx_lt = idx_sorted[mask_lt]
    count_lt = len(idx_lt)

    # Empatados al corte
    mask_eq = vals_sorted == cut_val
    idx_eq = idx_sorted[mask_eq]

    rng = check_random_state(seed)
    remaining = k - count_lt
    if remaining <= 0:
        chosen_eq = np.array([], dtype=idx_eq.dtype)
    else:
        if remaining >= len(idx_eq):
            chosen_eq = idx_eq
        else:
            chosen_eq = rng.choice(idx_eq, size=remaining, replace=False)

    chosen = set(idx_lt.tolist() + chosen_eq.tolist())

    y = pd.Series(0, index=range(len(values)), dtype=int)
    y.loc[list(chosen)] = 1
    return y

# ======================
# Splits estratificados
# ======================
def stratified_train_val_test(X: pd.DataFrame, y: pd.Series, train_ratio: float, val_ratio: float, seed: int = SEED):
    assert 0 < train_ratio < 1, "train debe ser (0,1)"
    assert 0 <= val_ratio < 1, "val debe ser [0,1)"
    assert train_ratio + val_ratio < 1, "train+val debe ser < 1"
    test_ratio = 1.0 - train_ratio - val_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    (train_idx, rest_idx) = next(sss1.split(X, y))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_rest, y_rest = X.iloc[rest_idx], y.iloc[rest_idx]

    # val sobre el resto
    val_share = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=val_share, random_state=seed)
    (val_idx_rel, test_idx_rel) = next(sss2.split(X_rest, y_rest))

    X_val, y_val = X_rest.iloc[val_idx_rel], y_rest.iloc[val_idx_rel]
    X_test, y_test = X_rest.iloc[test_idx_rel], y_rest.iloc[test_idx_rel]
    return X_train, y_train, X_val, y_val, X_test, y_test

# ======================
# Modelos y preprocesado
# ======================
def build_preprocessor(numeric_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preproc = ColumnTransformer(
        transformers=[("num", num_pipe, numeric_cols)],
        remainder="drop",
        sparse_threshold=0.0
    )
    return preproc

@dataclass
class RFConfig:
    algo: str = "rf"
    n_estimators: int = 2000           # máximo de árboles objetivo
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    class_weight: str = "balanced"
    random_state: int = SEED
    n_jobs: int = -1
    step: int = 100                   # nº de árboles añadidos en cada iteración
    patience: int = 20                # nº de iteraciones sin mejora para parar

@dataclass
class XGBConfig:
    algo: str = "xgb"
    n_estimators: int = 2000
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    reg_lambda: float = 1.0
    random_state: int = SEED
    n_jobs: int = -1
    early_stopping_rounds: int = 50
    verbose: bool = True              # mostrar progreso

def build_model(cfg: Any):
    if getattr(cfg, "algo", "rf") == "rf":
        model = RandomForestClassifier(
            n_estimators=cfg.step,            # se ajusta incrementalmente
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            warm_start=True,                  # permite añadir más árboles
        )
        return model
    elif getattr(cfg, "algo", "") == "xgb":
        if not _HAVE_XGB:
            raise RuntimeError("Has pedido XGBoost pero no está instalado. Ejecuta: pip install xgboost")
        model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            tree_method="hist",
            eval_metric="auc",
            enable_categorical=False,
            verbosity=1 if cfg.verbose else 0,
        )
        return model
    else:
        raise ValueError("Algoritmo no soportado")

def fit_rf_with_early_stopping(preproc: ColumnTransformer, cfg: RFConfig,
                               X_train, y_train, X_val, y_val):
    """
    Entrena RF incrementalmente, evaluando tras cada bloque 'step'.
    Parada cuando no mejora 'patience' iteraciones la AUC de validación.
    Devuelve el mejor modelo y el preprocesador.
    """
    # Ajustar preprocesador una sola vez
    X_tr = preproc.fit_transform(X_train)
    X_v  = preproc.transform(X_val)

    model = build_model(cfg)
    best_auc = -np.inf
    best_model = None
    best_iter = 0
    no_improve = 0
    total_trees = 0

    print(f"[INFO] Comienza entrenamiento RF con early stopping | step={cfg.step}, max_trees={cfg.n_estimators}, patience={cfg.patience}")

    while total_trees < cfg.n_estimators:
        # Aumentar nº de árboles hasta total_trees+step
        model.n_estimators = min(total_trees + cfg.step, cfg.n_estimators)
        model.fit(X_tr, y_train)

        total_trees = model.n_estimators
        # Evaluar
        prob_tr = model.predict_proba(X_tr)[:,1]
        prob_v  = model.predict_proba(X_v)[:,1]
        auc_tr  = roc_auc_score(y_train, prob_tr)
        auc_v   = roc_auc_score(y_val,  prob_v)
        f1_tr   = f1_score(y_train, (prob_tr>=0.5).astype(int))
        f1_v    = f1_score(y_val,    (prob_v>=0.5).astype(int))

        print(f"[PROGRESO] trees={total_trees:4d} | AUC train={auc_tr:.4f} val={auc_v:.4f} | F1 train={f1_tr:.4f} val={f1_v:.4f}")

        # Actualizar mejor
        if auc_v > best_auc + 1e-6:
            best_auc = auc_v
            best_model = copy.deepcopy(model)  # snapshot
            best_iter = total_trees
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print(f"[EARLY] Sin mejora en {cfg.patience} iteraciones. Paro en {total_trees} árboles. Mejor en {best_iter} (AUC val={best_auc:.4f}).")
                break

    if best_model is None:
        best_model = model
        best_iter = total_trees
        print(f"[INFO] No hubo mejora intermedia; uso el último modelo con {best_iter} árboles.")

    # Reemplazar n_estimators por best_iter para coherencia
    best_model.n_estimators = best_iter
    return best_model, preproc

def fit_xgb_with_logging(preproc: ColumnTransformer, cfg: XGBConfig,
                         X_train, y_train, X_val, y_val):
    X_tr = preproc.fit_transform(X_train)
    X_v  = preproc.transform(X_val)
    model = build_model(cfg)
    print(f"[INFO] Comienza entrenamiento XGB con early_stopping_rounds={cfg.early_stopping_rounds}")
    model.fit(
        X_tr, y_train,
        eval_set=[(X_v, y_val)],
        verbose=cfg.verbose,
        early_stopping_rounds=cfg.early_stopping_rounds
    )
    print(f"[INFO] Mejor iteración XGB: {model.best_iteration} | Mejor puntuación val (eval_metric): {model.best_score}")
    return model, preproc

def predict_proba(model, preproc, X):
    Xp = preproc.transform(X)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xp)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(Xp)
        z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        return z
    preds = model.predict(Xp)
    return preds.astype(float)

def evaluate_all(model, preproc, X, y, threshold: float = 0.5):
    prob = predict_proba(model, preproc, X)
    pred = (prob >= threshold).astype(int)
    out = {}
    out["roc_auc"] = roc_auc_score(y, prob)
    out["pr_auc"]  = average_precision_score(y, prob)
    out["f1"]      = f1_score(y, pred)
    out["precision"] = precision_score(y, pred, zero_division=0)
    out["recall"]  = recall_score(y, pred, zero_division=0)
    out["confusion_matrix"] = confusion_matrix(y, pred)
    out["y_prob"] = prob
    out["y_pred"] = pred
    return out

def plot_and_save_curves(model, preproc, X, y, out_dir: Path, prefix: str):
    prob = predict_proba(model, preproc, X)
    RocCurveDisplay.from_predictions(y, prob)
    plt.title(f"ROC - {prefix}")
    plt.tight_layout()
    plt.savefig(figure_path(out_dir, f"{prefix}_roc.png")); plt.close()

    PrecisionRecallDisplay.from_predictions(y, prob)
    plt.title(f"PR - {prefix}")
    plt.tight_layout()
    plt.savefig(figure_path(out_dir, f"{prefix}_pr.png")); plt.close()

def plot_confusion(cm: np.ndarray, out_dir: Path, prefix: str):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(f"Matriz de confusión - {prefix}")
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(figure_path(out_dir, f"{prefix}_cm.png"))
    plt.close(fig)

def top_feature_importances(model, feature_names: List[str], k: int = 20) -> List[Tuple[str, float]]:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif getattr(model, "get_booster", None) is not None:
        try:
            imp = model.get_booster().get_fscore()
            importances = np.zeros(len(feature_names))
            for key,score in imp.items():
                idx = int(key[1:])
                if 0 <= idx < len(feature_names):
                    importances[idx] = score
        except Exception:
            importances = None
    if importances is None:
        return []
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]

def try_save_shap(model, preproc, X_sample: pd.DataFrame, out_dir: Path, prefix: str):
    if not _HAVE_SHAP:
        return False
    try:
        Xp = preproc.transform(X_sample)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xp)
        shap.summary_plot(shap_values, Xp, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(figure_path(out_dir, f"{prefix}_shap_bar.png")); plt.close()
        return True
    except Exception:
        return False

# ======================
# Guardado / carga
# ======================
def save_all(model_name: str, model, preproc, config: Dict[str, Any], feature_names: List[str],
             metrics_val: Dict[str, Any], metrics_test: Dict[str, Any],
             splits_info: Dict[str, Any], data_hash: str, model_dir: Path):
    ensure_dir(model_dir)
    ckpt = model_dir / "checkpoints"
    ensure_dir(ckpt)
    plots = model_dir / "plots"
    ensure_dir(plots)

    joblib.dump(model, ckpt / "model.pkl")
    joblib.dump(preproc, ckpt / "preproc.pkl")
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Importancias top-K
    topk = top_feature_importances(model, feature_names, k=20)

    # Graficas
    plot_and_save_curves(model, preproc, splits_info["X_val"], splits_info["y_val"], plots, "val")
    plot_confusion(metrics_val["confusion_matrix"], plots, "val")
    plot_and_save_curves(model, preproc, splits_info["X_test"], splits_info["y_test"], plots, "test")
    plot_confusion(metrics_test["confusion_matrix"], plots, "test")

    # SHAP (opcional, si está instalado) — usar una muestra para no saturar
    _ = try_save_shap(model, preproc, splits_info["X_val"].sample(min(500, len(splits_info["X_val"])), random_state=SEED), plots, "val")

    # Guardar info
    info_lines = []
    info_lines.append(f"fecha: {now_str()}")
    info_lines.append(f"seed: {SEED}")
    info_lines.append(f"data_hash: {data_hash}")
    info_lines.append(f"n_features: {len(feature_names)}")
    info_lines.append(f"config: {json.dumps(config, ensure_ascii=False)}")
    info_lines.append("== metrics_val ==")
    for k,v in metrics_val.items():
        if k in ("y_prob","y_pred"): continue
        info_lines.append(f"{k}: {v}")
    info_lines.append("== metrics_test ==")
    for k,v in metrics_test.items():
        if k in ("y_prob","y_pred"): continue
        info_lines.append(f"{k}: {v}")
    if topk:
        info_lines.append("== top20_importances ==")
        for name,score in topk:
            info_lines.append(f"{name}: {score:.6f}")
    with open(model_dir / "model_info.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(info_lines))

    # Splits — IDs ya preparados
    splits_csv = model_dir / "splits.csv"
    df_splits = pd.DataFrame({
        "train_ids": pd.Series(splits_info["train_ids"]),
        "val_ids":   pd.Series(splits_info["val_ids"]),
        "test_ids":  pd.Series(splits_info["test_ids"]),
    })
    df_splits.to_csv(splits_csv, index=False)

def load_checkpoint(model_name: str) -> Tuple[Any, Any, Dict[str, Any]]:
    model_dir = MODELS_DIR / model_name
    ckpt = model_dir / "checkpoints"
    model = joblib.load(ckpt / "model.pkl")
    preproc = joblib.load(ckpt / "preproc.pkl")
    cfg = {}
    try:
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass
    return model, preproc, cfg

def exists_checkpoint(model_name: str) -> bool:
    ckpt = MODELS_DIR / model_name / "checkpoints"
    return (ckpt / "model.pkl").exists() and (ckpt / "preproc.pkl").exists()

def has_config(model_name: str) -> bool:
    return (MODELS_DIR / model_name / "config.json").exists()

# ======================
# Búsqueda de HP (test)
# ======================
def hyperparam_search(X_train, y_train, X_val, y_val, algo: str, seed: int = SEED):
    rng = check_random_state(seed)
    leaderboard = []

    if algo == "rf":
        space = {
            "n_estimators": [400, 800, 1200, 1600],
            "max_depth": [None, 8, 12, 16, 24],
            "min_samples_leaf": [1, 2, 4],
        }
        n_iter = 12
        configs = list(ParameterSampler(space, n_iter=n_iter, random_state=rng))
        for cfg in configs:
            cfg_obj = RFConfig(**{"algo":"rf", **cfg})
            preproc = build_preprocessor([c for c in X_train.columns])
            # Entrenamiento rápido sin incremental (para test)
            model = RandomForestClassifier(
                n_estimators=cfg_obj.n_estimators,
                max_depth=cfg_obj.max_depth,
                min_samples_leaf=cfg_obj.min_samples_leaf,
                class_weight=cfg_obj.class_weight,
                random_state=cfg_obj.random_state,
                n_jobs=cfg_obj.n_jobs,
            )
            X_tr = preproc.fit_transform(X_train); X_v = preproc.transform(X_val)
            model.fit(X_tr, y_train)
            prob_v = model.predict_proba(X_v)[:,1]; pred_v = (prob_v>=0.5).astype(int)
            auc = roc_auc_score(y_val, prob_v); f1 = f1_score(y_val, pred_v)
            leaderboard.append((auc, f1, cfg_obj, {"roc_auc":auc, "f1":f1}))
        leaderboard.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return leaderboard[0], leaderboard

    elif algo == "xgb":
        if not _HAVE_XGB:
            raise RuntimeError("XGBoost no está instalado. Instálalo con: pip install xgboost")
        space = {
            "n_estimators": [300, 600, 900, 1200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        }
        n_iter = 18
        configs = list(ParameterSampler(space, n_iter=n_iter, random_state=rng))
        for cfg in configs:
            cfg_obj = XGBConfig(**{"algo":"xgb", **cfg})
            preproc = build_preprocessor([c for c in X_train.columns])
            X_tr = preproc.fit_transform(X_train); X_v = preproc.transform(X_val)
            model = xgb.XGBClassifier(
                n_estimators=cfg_obj.n_estimators, max_depth=cfg_obj.max_depth,
                learning_rate=cfg_obj.learning_rate, subsample=cfg_obj.subsample,
                colsample_bytree=cfg_obj.colsample_bytree, reg_lambda=cfg_obj.reg_lambda,
                random_state=cfg_obj.random_state, n_jobs=cfg_obj.n_jobs,
                tree_method="hist", eval_metric="auc"
            )
            model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
            prob_v = model.predict_proba(X_v)[:,1]; pred_v = (prob_v>=0.5).astype(int)
            auc = roc_auc_score(y_val, prob_v); f1 = f1_score(y_val, pred_v)
            leaderboard.append((auc, f1, cfg_obj, {"roc_auc":auc, "f1":f1}))
        leaderboard.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return leaderboard[0], leaderboard

    else:
        raise ValueError("Algoritmo no soportado en test")

# ======================
# CLI
# ======================
def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Entrenador de modelos TFM (clasificación alta actividad).")
    p.add_argument("--dir", type=str, default=str(DEFAULT_DATA), help="Ruta al CSV de descriptores.")
    p.add_argument("--ic50", type=float, default=DEFAULT_IC50_FRAC, help="Fracción de activos (0<f<=1).")
    p.add_argument("--train", type=float, default=DEFAULT_TRAIN, help="Proporción de entrenamiento (por defecto 0.75).")
    p.add_argument("--val", type=float, default=DEFAULT_VAL, help="Proporción de validación (por defecto 0.15).")
    p.add_argument("--model", type=str, default="base", help="Modo: base|new|test|<nombre_modelo>")
    p.add_argument("--retrain", action="store_true", help="Forzar entrenamiento aunque exista checkpoint.")

    # Parámetros para --model new
    p.add_argument("--algo", type=str, choices=["rf","xgb"], default="rf", help="Algoritmo (solo en --model new).")
    p.add_argument("--name", type=str, default="", help="Nombre del modelo (carpeta en Models/, solo en --model new).")

    # Hiperparámetros RF
    p.add_argument("--rf_n_estimators", type=int, default=2500, help="Tope de árboles (RF).")
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--rf_min_samples_leaf", type=int, default=1)
    p.add_argument("--rf_step", type=int, default=100, help="Árboles añadidos por iteración (RF).")
    p.add_argument("--patience", type=int, default=15, help="Iteraciones sin mejora hasta parar (RF).")

    # Hiperparámetros XGB
    p.add_argument("--xgb_n_estimators", type=int, default=2000)
    p.add_argument("--xgb_max_depth", type=int, default=6)
    p.add_argument("--xgb_learning_rate", type=float, default=0.1)
    p.add_argument("--xgb_subsample", type=float, default=1.0)
    p.add_argument("--xgb_colsample_bytree", type=float, default=1.0)
    p.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    p.add_argument("--xgb_early_stopping", type=int, default=50, help="early_stopping_rounds de XGB.")
    p.add_argument("--xgb_verbose", action="store_true", help="Mostrar log detallado de XGB.")
    return p.parse_args()

def make_cfg_obj_from_args(args, algo: str):
    if algo == "rf":
        return RFConfig(
            algo="rf",
            n_estimators=args.rf_n_estimators,
            max_depth=(None if args.rf_max_depth in (None, 0) else args.rf_max_depth),
            min_samples_leaf=args.rf_min_samples_leaf,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            step=max(1, args.rf_step),
            patience=max(1, args.patience),
        )
    else:
        return XGBConfig(
            algo="xgb",
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=max(1, args.xgb_early_stopping),
            verbose=bool(args.xgb_verbose),
        )

# ======================
# Programa principal
# ======================
def main():
    set_global_seed(SEED)
    args = parse_args()

    data_path = Path(args.dir).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"No existe el CSV de entrada: {data_path}")

    print(f"[INFO] Cargando CSV: {data_path}")
    df = read_csv_robust(data_path)
    print(f"[INFO] CSV cargado. Filas: {len(df)}, Columnas: {len(df.columns)}")

    if "Standard Value" not in df.columns:
        raise ValueError("El CSV no contiene la columna 'Standard Value' necesaria para etiquetar por IC50.")

    df = df.copy()
    df["Standard Value"] = pd.to_numeric(df["Standard Value"], errors="coerce")
    df = df.dropna(subset=["Standard Value"]).reset_index(drop=True)

    # Etiquetado por fracción
    frac = float(args.ic50) if args.ic50 is not None else DEFAULT_IC50_FRAC
    if not (0.0 < frac <= 1.0):
        raise ValueError("--ic50 debe estar en (0,1].")
    y = build_label_by_fraction(df["Standard Value"], fraction=frac, seed=SEED)
    df["activo"] = y.astype(int)

    # Mostrar distribución y ejemplo de 'activo'
    vc = df["activo"].value_counts().sort_index()
    print(f"[INFO] Distribución de 'activo' (0/1): {dict(vc)}  (ic50={frac})")
    sample_cols = [c for c in ["Molecule ChEMBL ID","canonical_smiles","Smiles","Standard Value","activo"] if c in df.columns]
    print("[INFO] Vista de ejemplo con 'activo':")
    print(df[sample_cols].head(10))

    # Features: numéricas excepto meta prohibidas
    numeric_cols = [c for c in df.columns if c not in META_FORBIDDEN and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No se encontraron columnas numéricas de descriptores para entrenar.")
    print(f"[INFO] Nº de features numéricas: {len(numeric_cols)}")

    X = df[numeric_cols].copy()
    y = df["activo"].copy()

    # Splits
    train_ratio = float(args.train)
    val_ratio   = float(args.val)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_train_val_test(X, y, train_ratio, val_ratio, SEED)

    # Mostrar head del set de entrenamiento (incluyendo 'activo')
    print("[INFO] Head de entrenamiento (features + 'activo'):")
    with pd.option_context('display.max_columns', 12):
        Xy_train = X_train.copy()
        Xy_train["activo"] = y_train.values
        print(Xy_train.head(5))

    # IDs para guardar splits (tomados del df original)
    id_col = pick_id_column(df)
    if id_col in df.columns:
        train_ids = df.loc[X_train.index, id_col].astype(str).tolist()
        val_ids   = df.loc[X_val.index,   id_col].astype(str).tolist()
        test_ids  = df.loc[X_test.index,  id_col].astype(str).tolist()
    else:
        train_ids = [int(i) for i in X_train.index]
        val_ids   = [int(i) for i in X_val.index]
        test_ids  = [int(i) for i in X_test.index]

    splits_info = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "train_ids": train_ids,
        "val_ids":   val_ids,
        "test_ids":  test_ids,
    }

    # Selección de modo
    model_arg = args.model or "base"
    if model_arg in MODE_CHOICES:
        mode = model_arg
        model_name = DEFAULT_BASE_MODEL_NAME if mode == "base" else (args.name if mode == "new" else "test_tmp")
    else:
        mode = "named"
        model_name = model_arg

    prevent_sleep_start()
    try:
        data_hash = hash_file(data_path)

        if mode == "base":
            model_dir = MODELS_DIR / model_name
            ensure_dir(model_dir)
            if exists_checkpoint(model_name) and not args.retrain:
                print("[INFO] Cargando checkpoint existente (base). Usa --retrain para forzar nuevo entrenamiento.")
                model, preproc, loaded_cfg = load_checkpoint(model_name)
            else:
                # Entrenamiento base RF con early stopping
                cfg_obj = make_cfg_obj_from_args(args, "rf")
                preproc = build_preprocessor(numeric_cols)
                print(f"[INFO] Iniciando entrenamiento RF base con parámetros: {asdict(cfg_obj)}")
                model, preproc = fit_rf_with_early_stopping(preproc, cfg_obj,
                                                            splits_info["X_train"], splits_info["y_train"],
                                                            splits_info["X_val"], splits_info["y_val"])
                metrics_val  = evaluate_all(model, preproc, splits_info["X_val"],  splits_info["y_val"])
                metrics_test = evaluate_all(model, preproc, splits_info["X_test"], splits_info["y_test"])
                config_dict = {"algo": "rf", "rf": asdict(cfg_obj)}
                save_all(model_name, model, preproc, config_dict, numeric_cols, metrics_val, metrics_test, splits_info, data_hash, model_dir)
                print("[OK] Modelo base entrenado y guardado.")
                return 0

            # Incluso si cargamos, mostramos métricas actuales
            metrics_val  = evaluate_all(model, preproc, splits_info["X_val"],  splits_info["y_val"])
            metrics_test = evaluate_all(model, preproc, splits_info["X_test"], splits_info["y_test"])
            print(f"[INFO] Métricas (val): AUC={metrics_val['roc_auc']:.3f}, PR AUC={metrics_val['pr_auc']:.3f}, F1={metrics_val['f1']:.3f}")
            print(f"[INFO] Métricas (test): AUC={metrics_test['roc_auc']:.3f}, PR AUC={metrics_test['pr_auc']:.3f}, F1={metrics_test['f1']:.3f}")
            return 0

        elif mode == "new":
            if not args.name:
                raise ValueError("Debes indicar --name para --model new")
            algo = args.algo
            cfg_obj = make_cfg_obj_from_args(args, algo)
            preproc = build_preprocessor(numeric_cols)
            print(f"[INFO] Iniciando entrenamiento nuevo '{args.name}' con parámetros: {asdict(cfg_obj)}")

            if algo == "rf":
                model, preproc = fit_rf_with_early_stopping(preproc, cfg_obj,
                                                            splits_info["X_train"], splits_info["y_train"],
                                                            splits_info["X_val"], splits_info["y_val"])
            else:
                model, preproc = fit_xgb_with_logging(preproc, cfg_obj,
                                                      splits_info["X_train"], splits_info["y_train"],
                                                      splits_info["X_val"], splits_info["y_val"])

            metrics_val  = evaluate_all(model, preproc, splits_info["X_val"],  splits_info["y_val"])
            metrics_test = evaluate_all(model, preproc, splits_info["X_test"], splits_info["y_test"])
            model_dir = MODELS_DIR / args.name
            config_dict = {"algo": algo, algo: asdict(cfg_obj)}
            save_all(args.name, model, preproc, config_dict, numeric_cols, metrics_val, metrics_test, splits_info, data_hash, model_dir)
            print(f"[OK] Modelo '{args.name}' entrenado y guardado.")
            return 0

        elif mode == "test":
            print("[INFO] Iniciando búsqueda de hiperparámetros (test).")
            best_rf, board_rf = hyperparam_search(splits_info["X_train"], splits_info["y_train"],
                                                  splits_info["X_val"], splits_info["y_val"], "rf", SEED)
            print("== RESULTADOS RF (top 5) ==")
            for i, row in enumerate(board_rf[:5], start=1):
                auc, f1, cfg, metrics = row
                print(f"{i:02d}) AUC={auc:.4f}  F1={f1:.4f}  cfg={cfg}")

            if _HAVE_XGB:
                best_xgb, board_xgb = hyperparam_search(splits_info["X_train"], splits_info["y_train"],
                                                        splits_info["X_val"], splits_info["y_val"], "xgb", SEED)
                print("== RESULTADOS XGB (top 5) ==")
                for i, row in enumerate(board_xgb[:5], start=1):
                    auc, f1, cfg, metrics = row
                    print(f"{i:02d}) AUC={auc:.4f}  F1={f1:.4f}  cfg={cfg}")
            else:
                print("[AVISO] XGBoost no está instalado; se omite su búsqueda.")
            print("[INFO] Fin de test (no se guardan modelos).")
            return 0

        else:  # named
            model_dir = MODELS_DIR / model_name
            ensure_dir(model_dir)
            cfg_loaded = {}
            if has_config(model_name):
                with open(model_dir / "config.json", "r", encoding="utf-8") as f:
                    cfg_loaded = json.load(f)
            algo = (cfg_loaded.get("algo") or "rf").lower()
            if algo == "rf":
                rf_defaults = asdict(RFConfig())
                rf_cfg = rf_defaults | cfg_loaded.get("rf", {})
                cfg_obj = RFConfig(**rf_cfg)
            else:
                if not _HAVE_XGB:
                    raise RuntimeError("Config indica XGB pero no está instalado (pip install xgboost).")
                xgb_defaults = asdict(XGBConfig())
                xgb_cfg = xgb_defaults | cfg_loaded.get("xgb", {})
                cfg_obj = XGBConfig(**xgb_cfg)

            preproc = build_preprocessor(numeric_cols)

            if exists_checkpoint(model_name) and not args.retrain:
                print(f"[INFO] Cargando checkpoint existente ({model_name}). Usa --retrain para forzar nuevo entrenamiento.")
                model, preproc, _ = load_checkpoint(model_name)
            else:
                print(f"[INFO] Entrenando modelo '{model_name}' con configuración cargada: {asdict(cfg_obj)}")
                if algo == "rf":
                    model, preproc = fit_rf_with_early_stopping(preproc, cfg_obj,
                                                                splits_info["X_train"], splits_info["y_train"],
                                                                splits_info["X_val"], splits_info["y_val"])
                else:
                    model, preproc = fit_xgb_with_logging(preproc, cfg_obj,
                                                          splits_info["X_train"], splits_info["y_train"],
                                                          splits_info["X_val"], splits_info["y_val"])
                metrics_val  = evaluate_all(model, preproc, splits_info["X_val"],  splits_info["y_val"])
                metrics_test = evaluate_all(model, preproc, splits_info["X_test"], splits_info["y_test"])
                config_dict = {"algo": algo, algo: asdict(cfg_obj)}
                save_all(model_name, model, preproc, config_dict, numeric_cols, metrics_val, metrics_test, splits_info, data_hash, model_dir)
                print(f"[OK] Modelo '{model_name}' entrenado y guardado.")
                return 0

            metrics_val  = evaluate_all(model, preproc, splits_info["X_val"],  splits_info["y_val"])
            metrics_test = evaluate_all(model, preproc, splits_info["X_test"], splits_info["y_test"])
            print(f"[INFO] Métricas (val): AUC={metrics_val['roc_auc']:.3f}, PR AUC={metrics_val['pr_auc']:.3f}, F1={metrics_val['f1']:.3f}")
            print(f"[INFO] Métricas (test): AUC={metrics_test['roc_auc']:.3f}, PR AUC={metrics_test['pr_auc']:.3f}, F1={metrics_test['f1']:.3f}")
            return 0

    finally:
        prevent_sleep_stop()

if __name__ == "__main__":
    sys.exit(main())
