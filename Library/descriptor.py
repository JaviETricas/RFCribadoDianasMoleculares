#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
descriptor.py — Procesado robusto (sin argumentos) alineado con el CSV de referencia, pero
sin tirar todas las filas: sólo borra registros con valores inválidos en columnas "clave"
que estén soportadas por TU build de RDKit. Además:
- Usa rutas relativas al repo (REPO_ROOT = carpeta padre de Library).
- Selecciona el CSV más reciente en Dataset/Brutos.
- Filtra a IC50 en CHEMBL332 con unidades nM y relación '=' (normalizada).
- Calcula descriptores (core + Chi + VSA + MQN) con fallbacks.
- Deduplica por SMILES canónico.
- Elimina columnas descriptoras que queden 100% NaN (no soportadas en tu RDKit).
- Guarda SIEMPRE CSV como Dataset/Procesados/descriptores.csv.
"""

import os, sys, math
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import argparse


# =================== RDKit imports (seguros) ===================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen
    try:
        from rdkit.Chem import rdMolDescriptors as rdm
    except Exception:
        rdm = None
    _HAVE_RDKIT = True
    try:
        import rdkit as _rdkit_pkg
        _rdkit_version = getattr(_rdkit_pkg, "__version__", "desconocida")
    except Exception:
        _rdkit_version = "desconocida"
except ImportError:
    _HAVE_RDKIT = False
    _rdkit_version = "no-importado"

# =================== Rutas relativas ===================
THIS_FILE   = Path(__file__).resolve()
LIB_DIR     = THIS_FILE.parent
REPO_ROOT   = LIB_DIR.parent

DATASET_DIR = REPO_ROOT / "Dataset"
BRUTOS_DIR  = DATASET_DIR / "Brutos"
PROCES_DIR  = DATASET_DIR / "Procesados"
PRED_DIR    = DATASET_DIR / "Prediccion"
OUT_NAME    = "descriptores.csv"

# ===== Meta a preservar si existen =====
META_COLS_KEEP = [
    "Molecule ChEMBL ID","Smiles","Standard Type","Standard Relation",
    "Standard Value","Standard Units","Target ChEMBL ID"
]

# ===== Descriptor columns objetivo (nombres REFERENCE) =====
BASE_CORE = ["SlogP","SMR","LabuteASA","TPSA","AMW","ExactMW",
             "NumLipinskiHBA","NumLipinskiHBD","NumRotatableBonds","FractionCSP3","HallKierAlpha"]
CHI      = ["Chi0v","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v"]
PEOE_VSA = [f"peoe_VSA{i}" for i in range(1,15)]
SLOGP_VSA= [f"slogp_VSA{i}" for i in range(1,13)]
SMR_VSA  = [f"smr_VSA{i}" for i in range(1,11)]
MQN      = [f"MQN{i}" for i in range(1,43)]

DESCRIPTOR_COLS = BASE_CORE + CHI + PEOE_VSA + SLOGP_VSA + SMR_VSA + MQN

# ===== Utilidades =====
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def find_latest_csv(brutos_dir: Path) -> Path:
    if not brutos_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {brutos_dir}")
    csvs = [p for p in brutos_dir.iterdir() if p.is_file() and p.suffix.lower()==".csv"]
    if not csvs:
        raise FileNotFoundError(f"No se encontraron CSV en: {brutos_dir}")
    csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0]

def read_csv_autodetect(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")

def detect_smiles_col(df: pd.DataFrame) -> str:
    for c in ["Smiles","smiles","SMILES","canonical_smiles","CanSmiles"]:
        if c in df.columns: return c
    for c in df.columns:
        if "smile" in str(c).lower():
            return c
    raise ValueError("No se encontró columna 'Smiles' o similar. Columnas: "+", ".join(map(str,df.columns)))

def smiles_to_mol(smiles: str, sanitize: bool = True):
    if not isinstance(smiles,str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles, sanitize=sanitize)
    except Exception:
        return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# ====== Cálculos de descriptores (con fallbacks) ======
def calc_core_descriptors(mol: Chem.Mol) -> dict:
    out = {}
    # Crippen props
    out["SlogP"] = safe_float(Crippen.MolLogP(mol))
    out["SMR"]   = safe_float(Crippen.MolMR(mol))
    # TPSA
    if rdm and hasattr(rdm, "CalcTPSA"):
        out["TPSA"] = safe_float(rdm.CalcTPSA(mol))
    else:
        out["TPSA"] = safe_float(Descriptors.TPSA(mol))
    # LabuteASA
    if rdm and hasattr(rdm, "CalcLabuteASA"):
        out["LabuteASA"] = safe_float(rdm.CalcLabuteASA(mol))
    else:
        out["LabuteASA"] = np.nan
    # Pesos
    out["AMW"] = safe_float(Descriptors.MolWt(mol))
    if rdm and hasattr(rdm, "CalcExactMolWt"):
        out["ExactMW"] = safe_float(rdm.CalcExactMolWt(mol))
    else:
        ex = getattr(Descriptors, "ExactMolWt", None)
        out["ExactMW"] = safe_float(ex(mol)) if callable(ex) else np.nan
    # NumLipinskiHBA
    hba = None
    try:
        from rdkit.Chem import Lipinski as _Lip
        f = getattr(_Lip, "NumHBA", None)
        hba = f(mol) if callable(f) else None
    except Exception:
        hba = None
    if hba is None and rdm and hasattr(rdm, "CalcNumHBA"):
        try: hba = rdm.CalcNumHBA(mol)
        except Exception: hba = None
    if hba is None:
        fn = getattr(Descriptors, "NumHAcceptors", None)
        hba = fn(mol) if callable(fn) else np.nan
    out["NumLipinskiHBA"] = safe_float(hba)
    # NumLipinskiHBD
    hbd = None
    try:
        from rdkit.Chem import Lipinski as _Lip
        f = getattr(_Lip, "NumHBD", None)
        hbd = f(mol) if callable(f) else None
    except Exception:
        hbd = None
    if hbd is None and rdm and hasattr(rdm, "CalcNumHBD"):
        try: hbd = rdm.CalcNumHBD(mol)
        except Exception: hbd = None
    if hbd is None:
        fn = getattr(Descriptors, "NumHDonors", None)
        hbd = fn(mol) if callable(fn) else np.nan
    out["NumLipinskiHBD"] = safe_float(hbd)
    # Rotables, fracción sp3, HallKierAlpha
    if rdm and hasattr(rdm, "CalcNumRotatableBonds"):
        out["NumRotatableBonds"] = safe_float(rdm.CalcNumRotatableBonds(mol))
    else:
        out["NumRotatableBonds"] = safe_float(Descriptors.NumRotatableBonds(mol))
    out["FractionCSP3"]      = safe_float(Descriptors.FractionCSP3(mol))
    out["HallKierAlpha"]     = safe_float(Descriptors.HallKierAlpha(mol))
    return out

def calc_chi_descriptors(mol: Chem.Mol) -> dict:
    names = ["Chi0v","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v"]
    funcs = [Descriptors.Chi0v,Descriptors.Chi1n,Descriptors.Chi1v,Descriptors.Chi2n,Descriptors.Chi2v,Descriptors.Chi3n,Descriptors.Chi3v,Descriptors.Chi4n,Descriptors.Chi4v]
    return {n: safe_float(f(mol)) for n,f in zip(names,funcs)}

def _try_vector_func(module, name, count, mol, desired_prefix=None):
    out = {}
    prefix = desired_prefix if desired_prefix is not None else name
    if module and hasattr(module, name):
        try:
            vec = getattr(module, name)
            vals = vec(mol)  # e.g., rdm.PEOE_VSA(mol) -> iterable
            for i, v in enumerate(vals, start=1):
                out[f"{prefix}{i}"] = safe_float(v)
            return out
        except Exception:
            pass
    # Fallback a funciones individuales de Descriptors
    for i in range(1, count+1):
        fn = getattr(Descriptors, f"{name}{i}", None)
        out[f"{prefix}{i}"] = safe_float(fn(mol)) if callable(fn) else np.nan
    return out

def calc_vsa_blocks(mol: Chem.Mol) -> dict:
    out = {}
    out.update(_try_vector_func(rdm, "PEOE_VSA", 14, mol, "peoe_VSA"))
    out.update(_try_vector_func(rdm, "SlogP_VSA", 12, mol, "slogp_VSA"))
    out.update(_try_vector_func(rdm, "SMR_VSA", 10, mol, "smr_VSA"))
    return out

def calc_mqn(mol: Chem.Mol) -> dict:
    if rdm and hasattr(rdm, "CalcMQNs"):
        try:
            mqn = rdm.CalcMQNs(mol)
            return {f"MQN{i+1}": safe_float(v) for i, v in enumerate(mqn)}
        except Exception:
            pass
    # Fallback: todos NaN si no disponible en tu build
    return {f"MQN{i}": np.nan for i in range(1,43)}

# ====== Pre-filtro estilo referencia ======
def normalize_relation(val: str) -> str:
    # Quita comillas simples/dobles y espacios
    if val is None:
        return ""
    s = str(val).strip().strip("\"'").strip()
    return s

def filter_like_reference(df: pd.DataFrame) -> pd.DataFrame:
    cond = pd.Series([True]*len(df))
    if "Target ChEMBL ID" in df.columns:
        cond &= (df["Target ChEMBL ID"].astype(str) == "CHEMBL332")
    if "Standard Type" in df.columns:
        cond &= (df["Standard Type"].astype(str).str.upper() == "IC50")
    if "Standard Units" in df.columns:
        cond &= (df["Standard Units"].astype(str).str.lower() == "nm")
    if "Standard Relation" in df.columns:
        rel = df["Standard Relation"].apply(normalize_relation)
        cond &= (rel == "=")
    if "Standard Value" in df.columns:
        vals = pd.to_numeric(df["Standard Value"], errors="coerce")
        cond &= vals.notna() & (vals > 0)
    return df[cond].copy()

# ====== Pipeline ======
def process_file(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    # 0) Filtro estilo referencia (IC50, nM, '=' y CHEMBL332)
    df = filter_like_reference(df)

    # 1) Calcular descriptores
    rows = []
    for _, row in df.iterrows():
        base: Dict[str, object] = {}
        # Meta
        for m in META_COLS_KEEP:
            if m in row.index:
                base[m] = row[m]
        # Mol & canonical
        mol = smiles_to_mol(row[smiles_col] if smiles_col in row.index else None, sanitize=True)
        if mol is None:
            # Si no se puede parsear, dejamos NaN en todo y seguimos; se eliminará por columnas requeridas
            for c in DESCRIPTOR_COLS:
                base[c] = np.nan
            base["canonical_smiles"] = None
        else:
            try:
                base["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                base["canonical_smiles"] = None
            base.update(calc_core_descriptors(mol))
            base.update(calc_chi_descriptors(mol))
            base.update(calc_vsa_blocks(mol))
            base.update(calc_mqn(mol))
        rows.append(base)

    out = pd.DataFrame(rows)

    # 2) Deduplicado por canonical_smiles
    if "canonical_smiles" in out.columns:
        before = len(out)
        out = out.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)
        after = len(out)
        print(f"[INFO] Duplicados eliminados (canonical_smiles): {before - after}")

    # 3) Asegurar columnas objetivo y ordenar
    for c in DESCRIPTOR_COLS:
        if c not in out.columns:
            out[c] = np.nan
    cols = [c for c in META_COLS_KEEP if c in out.columns] + ["canonical_smiles"] + DESCRIPTOR_COLS
    out = out[cols]

    # 4) Determinar columnas *soportadas* por esta build (no 100% NaN)
    supported = [c for c in DESCRIPTOR_COLS if not out[c].isna().all()]
    unsupported = [c for c in DESCRIPTOR_COLS if out[c].isna().all()]
    if unsupported:
        print(f"[AVISO] Columnas no soportadas por tu RDKit (se mantienen fuera del filtro estricto): {', '.join(unsupported[:10])}{'...' if len(unsupported)>10 else ''}")

    # 5) Definir columnas *clave* para filtrar filas: core + chi + vsa que estén soportadas
    required_base = [c for c in BASE_CORE if c in supported]
    required_chi  = [c for c in CHI if c in supported]
    required_vsa  = [c for c in (PEOE_VSA + SLOGP_VSA + SMR_VSA) if c in supported]
    # MQN no lo forzamos: si tu build no lo soporta, no debe eliminar filas
    required_cols = required_base + required_chi + required_vsa

    # 6) Filtrar filas sólo si faltan valores en columnas *clave* soportadas
    before_rows = len(out)
    out_clean = out.dropna(subset=required_cols, how="any").reset_index(drop=True)
    after_rows = len(out_clean)
    print(f"[INFO] Filas antes del filtrado por NA en columnas clave: {before_rows}")
    print(f"[INFO] Filas tras el filtrado por NA en columnas clave:  {after_rows}")

    # 7) Si alguna columna descriptora quedó 100% NaN, puedes optar por eliminarla del output
    #    para acercarte más al CSV de referencia. Lo hacemos para que el fichero sea "usable".
    drop_allnan_cols = [c for c in DESCRIPTOR_COLS if out_clean[c].isna().all()]
    if drop_allnan_cols:
        out_clean = out_clean.drop(columns=drop_allnan_cols)
        print(f"[AVISO] Columnas eliminadas del CSV final por ser 100% NaN: {', '.join(drop_allnan_cols[:10])}{'...' if len(drop_allnan_cols)>10 else ''}")

    return out_clean

def parse_args():
    p = argparse.ArgumentParser(description="Genera descriptores (modo normal o predicción).")
    p.add_argument(
        "--predict",
        action="store_true",
        help="Si se indica, guarda en Dataset/Prediccion/descriptores.csv en lugar de Procesados."
    )
    return p.parse_args()

def main():
    if not _HAVE_RDKIT:
        sys.stderr.write(
            "[ERROR] No se pudo importar RDKit.\n"
            "Instálalo en Windows (PowerShell):\n"
            "  python -m venv .venv\n"
            "  .venv\\Scripts\\Activate.ps1\n"
            "  pip install rdkit pandas numpy\n"
        )
        return 1

    print(f"[INFO] RDKit versión { _rdkit_version }.")
    args = parse_args()
    print(f"[INFO] REPO_ROOT   : {REPO_ROOT}")
    print(f"[INFO] Dataset dir : {DATASET_DIR}")
    print(f"[INFO] Brutos dir  : {BRUTOS_DIR}")
    print(f"[INFO] Procesados  : {PROCES_DIR}")

    src = find_latest_csv(BRUTOS_DIR)
    print(f"[INFO] CSV seleccionado: {src}")
    df = read_csv_autodetect(src)
    smiles_col = detect_smiles_col(df)
    print(f"[INFO] Columna SMILES detectada: {smiles_col}")

    out_df = process_file(df, smiles_col=smiles_col)

    out_base = PRED_DIR if args.predict else PROCES_DIR
    ensure_dirs(out_base)
    dst = out_base / OUT_NAME
    out_df.to_csv(dst, index=False)
    print(f"[OK] Guardado CSV: {dst} (filas: {len(out_df)}, columnas: {len(out_df.columns)})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
