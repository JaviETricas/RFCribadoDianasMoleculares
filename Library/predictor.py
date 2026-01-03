#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import datetime
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

# Rutas relativas
THIS_FILE = Path(__file__).resolve()
LIB_DIR   = THIS_FILE.parent
REPO_ROOT = LIB_DIR.parent

DATASET_DIR    = REPO_ROOT / "Dataset"
PRED_DIR       = DATASET_DIR / "Prediccion"
DEFAULT_CSV    = PRED_DIR / "descriptores.csv"

MODELS_DIR     = REPO_ROOT / "Models"
RESULTS_DIR    = REPO_ROOT / "Resultados"

# Config / Constantes
SEED = 1234
DEFAULT_MODEL_NAME = "base"
DEFAULT_THRESHOLD  = 0.5

# Columnas meta a no usar como features
META_FORBIDDEN = {
    "Standard Value", "Molecule ChEMBL ID", "Smiles", "canonical_smiles",
    "Standard Type", "Standard Relation", "Standard Units", "Target ChEMBL ID", "activo",
    "PAINS_flag", "PAINS_terms", "Chelator_flag", "Chelator_terms",
    "Lipinski_violations", "Veber_violations",
}

FILTER_COLS = [
    "PAINS_flag", "PAINS_terms", "Chelator_flag", "Chelator_terms",
    "Lipinski_violations", "Veber_violations"
]

# Utilidades
def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")

def pick_id_column(df: pd.DataFrame) -> str:
    for c in ["Molecule ChEMBL ID", "canonical_smiles", "Smiles"]:
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
    if s in {"1", "true", "yes", "si", "sí", "y"}:
        return True
    try:
        return float(s) != 0.0
    except Exception:
        return bool(s)

def build_filters_summary(row: pd.Series) -> str:
    msgs = []

    def _clean_text(x) -> str:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
        s = str(x).strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        # Evitar separadores conflictivos o saltos de linea
        s = s.replace("\r", " ").replace("\n", " ").strip()
        return s

    # PAINS
    if "PAINS_flag" in row and truthy(row["PAINS_flag"]):
        term = _clean_text(row.get("PAINS_terms", ""))
        msgs.append(f"PAINS: {term if term else 'si'}")

    # Quelacion (queladores)
    if "Chelator_flag" in row and truthy(row["Chelator_flag"]):
        term = _clean_text(row.get("Chelator_terms", ""))
        msgs.append(f"Quelacion: {term if term else 'si'}")

    # Lipinski / Veber (si existen)
    lv = row.get("Lipinski_violations", np.nan)
    vv = row.get("Veber_violations", np.nan)

    try:
        if pd.notna(lv) and int(lv) > 0:
            msgs.append(f"Lipinski alertas: {int(lv)}")
    except Exception:
        pass

    try:
        if pd.notna(vv) and int(vv) > 0:
            msgs.append(f"Veber alertas: {int(vv)}")
    except Exception:
        pass

    return " | ".join(msgs) if msgs else "Sin alertas"

def _excel_engine_available() -> Optional[str]:
    # Pandas necesita un motor: openpyxl o xlsxwriter
    for eng in ("openpyxl", "xlsxwriter"):
        try:
            __import__(eng)
            return eng
        except Exception:
            continue
    return None

# CLI
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


# Principal
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
    ids = df[id_col] if id_col in df.columns else df.index

    resumen = pd.DataFrame({
        "id": ids,
        "prob_activo": y_prob,
        "activo_pred": y_pred,
        "_idx": df.index,  # para reordenar df a la vez que el resumen
    })

    # Añadir columna textual 'filtros' si existen columnas de filtros
    has_any_filters = any(c in df.columns for c in FILTER_COLS)
    if has_any_filters:
        resumen["filtros"] = [build_filters_summary(row) for _, row in df.iterrows()]
    else:
        resumen["filtros"] = ""

    # ORDENAR (mejor → peor): Activos primero y dentro por prob desc; luego inactivos por prob desc
    resumen = resumen.sort_values(by=["activo_pred", "prob_activo"], ascending=[False, False]).reset_index(drop=True)

    # Reordenar df acorde
    df_ord = df.reindex(resumen["_idx"]).reset_index(drop=True)

    # Construir tabla de presentacion 
    tabla_pred = resumen[["id", "prob_activo", "activo_pred", "filtros"]].copy()
    tabla_pred.insert(0, "rank", np.arange(1, len(tabla_pred) + 1))
    tabla_pred["prediccion"] = np.where(tabla_pred["activo_pred"] == 1, "Activo", "Inactivo")
    tabla_pred = tabla_pred[["rank", "id", "prob_activo", "prediccion", "filtros"]].rename(
        columns={
            "rank": "Rank",
            "id": "ID",
            "prob_activo": "Prob_activo",
            "prediccion": "Prediccion",
            "filtros": "Filtros",
        }
    )

    # Mostrar en pantalla — TOP 10
    top_n = 10
    print(f"[INFO] Predicciones (TOP {top_n}) — ordenadas mejor→peor:")
    with pd.option_context("display.max_rows", top_n, "display.max_columns", None, "display.max_colwidth", None, "display.width", 220):
        print(tabla_pred.head(top_n))

    # Guardar tabla de TODAS las predicciones
    # - Se guarda en la carpeta Resultados/
    # - Nombre: resultados_<fechaa>.csv
    ensure_dir(RESULTS_DIR)
    fecha = datetime.date.today().strftime("%Y%m%d")
    out_csv = RESULTS_DIR / f"resultados_{fecha}.csv"

    tabla_pred.to_csv(out_csv, index=False, sep=";", encoding="utf-8-sig", float_format="%.6f")
    print(f"[OK] CSV de predicciones guardado en: {out_csv} (filas={len(tabla_pred)})")



    # Informe Excel
    # Columnas de filtros (si faltan, se asumen “No”)
    pains_flag = df_ord["PAINS_flag"] if "PAINS_flag" in df_ord.columns else pd.Series([0]*len(df_ord))
    pains_terms = df_ord["PAINS_terms"] if "PAINS_terms" in df_ord.columns else pd.Series([""]*len(df_ord))
    chel_flag = df_ord["Chelator_flag"] if "Chelator_flag" in df_ord.columns else pd.Series([0]*len(df_ord))
    chel_terms = df_ord["Chelator_terms"] if "Chelator_terms" in df_ord.columns else pd.Series([""]*len(df_ord))

    # ID Molecula (ChEMBL/SMILES)
    chembl_col = "Molecule ChEMBL ID" if "Molecule ChEMBL ID" in df_ord.columns else None
    smiles_col = "canonical_smiles" if "canonical_smiles" in df_ord.columns else None

    def _fmt_id(i):
        chembl = str(df_ord.loc[i, chembl_col]).strip() if chembl_col else str(resumen.loc[i, "id"]).strip()
        smi = str(df_ord.loc[i, smiles_col]).strip() if smiles_col else ""
        if smi and smi.lower() != "nan":
            return f"{chembl} ({smi})"
        return chembl

    # Alertas legibles
    def _fmt_pains(i):
        if truthy(pains_flag.iloc[i]):
            t = str(pains_terms.iloc[i]).strip()
            if t and t.lower() != "nan":
                return f"Sí ({t})"
            return "Sí"
        return "No"

    def _fmt_chel(i):
        if truthy(chel_flag.iloc[i]):
            t = str(chel_terms.iloc[i]).strip()
            if t and t.lower() != "nan":
                return f"Sí ({t})"
            return "Sí"
        return "No"

    # Decisión final
    def _decision(i):
        activo = int(resumen.loc[i, "activo_pred"]) == 1
        has_pains = truthy(pains_flag.iloc[i])
        has_chel = truthy(chel_flag.iloc[i])

        if activo and has_pains:
            return "❌ Descartado"
        if activo and has_chel:
            return "⚠️ Revisión (Riesgo)"
        if activo:
            return "✅ Seleccionado"
        return "No seleccionado"

    informe = pd.DataFrame({
        "ID Molécula (ChEMBL/SMILES)": [_fmt_id(i) for i in range(len(resumen))],
        "Probabilidad (Score)": np.round(resumen["prob_activo"].astype(float).values, 3),
        "Predicción Clase": np.where(resumen["activo_pred"].values == 1, "Activo", "Inactivo"),
        "Alerta PAINS": [_fmt_pains(i) for i in range(len(resumen))],
        "Alerta Quelante": [_fmt_chel(i) for i in range(len(resumen))],
        "Decisión Final": [_decision(i) for i in range(len(resumen))],
    })

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = RESULTS_DIR / f"informe_resultados_{args.model}_{ts}.xlsx"
    engine = _excel_engine_available()

    if engine is None:
        print("[WARN] No se pudo generar Excel porque falta un motor (openpyxl/xlsxwriter).")
        print("       Instala uno en tu venv, por ejemplo: pip install openpyxl")
    else:
        with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
            informe.to_excel(writer, index=False, sheet_name="Informe")
            ws = writer.sheets["Informe"]

            # Congelar encabezado y ajustar anchos
            try:
                if engine == "xlsxwriter":
                    ws.freeze_panes(1, 0)
                    for col_idx, col_name in enumerate(informe.columns):
                        width = max(16, min(75, int(informe[col_name].astype(str).map(len).quantile(0.95)) + 2))
                        ws.set_column(col_idx, col_idx, width)
                else:  # openpyxl
                    ws.freeze_panes = "A2"
                    from openpyxl.utils import get_column_letter
                    for col_idx, col_name in enumerate(informe.columns, start=1):
                        col_letter = get_column_letter(col_idx)
                        width = max(16, min(75, int(informe[col_name].astype(str).map(len).quantile(0.95)) + 2))
                        ws.column_dimensions[col_letter].width = width
            except Exception:
                pass

        print(f"[OK] Informe Excel guardado en: {xlsx_path}")

    # Resumen corto por consola
    n = len(resumen)
    n1 = int((resumen["activo_pred"] == 1).sum())
    n0 = n - n1
    print(f"[RESUMEN] n={n} | activos_pred=1: {n1} | inactivos_pred=0: {n0} | umbral={thr}")
    if has_any_filters:
        flagged = resumen["filtros"].fillna("").str.strip().ne("Sin alertas")
        print(f"[RESUMEN] Con alguna alerta de filtros: {int(flagged.sum())} de {n}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
