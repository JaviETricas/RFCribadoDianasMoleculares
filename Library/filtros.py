#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, json
from pathlib import Path
import pandas as pd

# ==== RUTAS RELATIVAS AL REPO ====
THIS_FILE = Path(__file__).resolve()
LIB_DIR   = THIS_FILE.parent
REPO_ROOT = LIB_DIR.parent
DEFAULT_CSV = REPO_ROOT / "Dataset" / "Procesados" / "descriptores.csv"
PREDICTION_CSV = REPO_ROOT / "Dataset" / "Prediccion" / "descriptores.csv"


# ==== RDKit ====
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, FilterCatalog
except Exception as e:
    print("[ERROR] No se pudo importar RDKit.\n"
          "Instálalo en Windows (PowerShell) dentro de tu venv:\n"
          "  python -m venv .venv\n"
          "  .venv\\Scripts\\Activate.ps1\n"
          "  pip install rdkit pandas numpy\n")
    sys.exit(1)

# ==== ARGUMENTOS ====
import argparse
p = argparse.ArgumentParser(description="Aplicar filtros a descriptores.csv")
p.add_argument("--dir", type=str, default=None,
               help="CSV de descriptores a filtrar (por defecto Dataset/Procesados/descriptores.csv)")
p.add_argument("--predict", action="store_true", help="Usar Dataset/Prediccion/descriptores.csv como entrada/salida por defecto.")
p.add_argument("--all", action="store_true", help="Aplicar todos los filtros")
p.add_argument("--pains", action="store_true", help="Aplicar filtro PAINS")
p.add_argument("--chelators", action="store_true", help="Aplicar filtro de quelación pan-metalo")
p.add_argument("--druglikeness", action="store_true", help="Añadir nº de violaciones Lipinski/Veber")
args = p.parse_args()

if not any([args.all, args.pains, args.chelators, args.druglikeness]):
    print("[INFO] No se seleccionó ningún filtro (usa --all o alguno de --pains/--chelators/--druglikeness).")
    sys.exit(0)

if args.dir:
    csv_path = Path(args.dir).resolve()
else:
    csv_path = PREDICTION_CSV if args.predict else DEFAULT_CSV

if not csv_path.exists():
    print(f"[ERROR] No existe el CSV: {csv_path}")
    sys.exit(1)


# ==== CARGA CSV ====
def read_csv_robust(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")

df = read_csv_robust(csv_path)
print(f"[INFO] CSV cargado: {csv_path}  (filas={len(df)}, cols={len(df.columns)})")

# ==== DETECTAR COLUMNA SMILES ====
SMILES_CANDIDATES = ["canonical_smiles", "Smiles", "SMILES", "smiles"]
smiles_col = None
for c in SMILES_CANDIDATES:
    if c in df.columns:
        smiles_col = c
        break
if smiles_col is None:
    print("[ERROR] No se encuentra columna SMILES (canonical_smiles/Smiles/SMILES/smiles). "
          "Asegúrate de haber generado 'descriptores.csv' con descriptor.py.")
    sys.exit(1)

# ==== PREPARAR MÓLÉCULAS ====
def to_mol(smiles: str):
    if pd.isna(smiles): 
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None

mols = [to_mol(s) for s in df[smiles_col].tolist()]

# ==== 1) PAINS ====
def apply_pains(mols):
    params = FilterCatalog.FilterCatalogParams()
    # Catálogo completo de PAINS
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    pains_flag = []
    pains_terms = []

    for mol in mols:
        if mol is None:
            pains_flag.append(0)
            pains_terms.append("")
            continue
        matches = catalog.GetMatches(mol)  # lista de FilterMatch
        if matches:
            pains_flag.append(1)
            # obtener descripciones de cada entrada que hace match
            terms = []
            for m in matches:
                try:
                    entry = m.GetFilterEntry()
                    terms.append(entry.GetDescription())
                except Exception:
                    terms.append("PAINS_match")
            pains_terms.append("|".join(sorted(set(terms))))
        else:
            pains_flag.append(0)
            pains_terms.append("")
    return pains_flag, pains_terms

# ==== 2) QUELACIÓN PAN-METALO (SMARTS) ====
# Nota: patrones aproximados y conservadores; pueden ajustarse
CHELATOR_SMARTS = {
    "catechol_o_dihydroxybenzene": "c1c(O)cccc1O",           # aproximado (orto-diOH aromático)
    "vicinal_diol":                "[CX4](O)[CX4](O)",        # 1,2-diol alifático genérico
    "hydroxamate":                 "[CX3](=O)N[OX2H,OX1-]",   # -C(=O)-N-O(H/-)
    "dithiocarbamate":             "[NX3][CX3](=S)[SX2]",     # -N-C(=S)-S-
    "thiol":                       "[SX2H]",                  # -SH
    "8_hydroxyquinoline":          "Oc1cccc2ncccc12",         # 8-hidroxiquinolina
    # Se puedes añadir más: hydroxypyridinone, 2,3-dihidroxipiridina, etc.
}
CHELATOR_PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in CHELATOR_SMARTS.items()}

def apply_chelators(mols):
    che_flag = []
    che_terms = []
    for mol in mols:
        if mol is None:
            che_flag.append(0)
            che_terms.append("")
            continue
        hits = []
        for name, patt in CHELATOR_PATTERNS.items():
            if patt is None:
                continue
            try:
                if mol.HasSubstructMatch(patt):
                    hits.append(name)
            except Exception:
                pass
        if hits:
            che_flag.append(1)
            che_terms.append("|".join(sorted(set(hits))))
        else:
            che_flag.append(0)
            che_terms.append("")
    return che_flag, che_terms

# ==== 3) DRUG-LIKENESS (Lipinski/Veber) ====
def lipinski_violations(mol):
    """Cuenta violaciones de Lipinski: HBA<=10, HBD<=5, MW<=500, logP<=5."""
    if mol is None:
        return None
    HBA = rdMolDescriptors.CalcNumHBA(mol)
    HBD = rdMolDescriptors.CalcNumHBD(mol)
    MW  = Descriptors.MolWt(mol)
    logP = Crippen.MolLogP(mol)
    v = 0
    if HBA > 10: v += 1
    if HBD > 5:  v += 1
    if MW  > 500: v += 1
    if logP > 5: v += 1
    return v

def veber_violations(mol):
    """Cuenta violaciones de Veber (TPSA<=140 o HBD+HBA<=12) y rotables<=10."""
    if mol is None:
        return None
    TPSA = rdMolDescriptors.CalcTPSA(mol)
    HBA  = rdMolDescriptors.CalcNumHBA(mol)
    HBD  = rdMolDescriptors.CalcNumHBD(mol)
    ROT  = rdMolDescriptors.CalcNumRotatableBonds(mol)
    cond1 = (TPSA <= 140) or ((HBA + HBD) <= 12)
    cond2 = (ROT <= 10)
    v = 0
    if not cond1: v += 1
    if not cond2: v += 1
    return v

def apply_druglikeness(mols):
    lip_v = []
    veb_v = []
    for mol in mols:
        lip_v.append(lipinski_violations(mol))
        veb_v.append(veber_violations(mol))
    return lip_v, veb_v

# ==== APLICAR SEGÚN FLAGS ====
did_any = False

if args.all or args.pains:
    pains_flag, pains_terms = apply_pains(mols)
    df["PAINS_flag"]  = pains_flag
    df["PAINS_terms"] = pains_terms
    did_any = True
    print(f"[OK] PAINS aplicado. Flag=1 en {int(sum(pains_flag))} compuestos.")

if args.all or args.chelators:
    che_flag, che_terms = apply_chelators(mols)
    df["Chelator_flag"]  = che_flag
    df["Chelator_terms"] = che_terms
    did_any = True
    print(f"[OK] Quelación aplicada. Flag=1 en {int(sum(1 for x in che_flag if x==1))} compuestos.")

if args.all or args.druglikeness:
    lip_v, veb_v = apply_druglikeness(mols)
    df["Lipinski_violations"] = lip_v   # 0–4 (None si SMILES inválido)
    df["Veber_violations"]    = veb_v   # 0–2 (None si SMILES inválido)
    did_any = True
    # Resumen
    l_ok  = sum(1 for v in lip_v if v is not None and v == 0)
    v_ok  = sum(1 for v in veb_v if v is not None and v == 0)
    print(f"[OK] Drug-likeness: Lipinski ok (0 violaciones) en {l_ok}; Veber ok (0 violaciones) en {v_ok} compuestos.")

if not did_any:
    print("[INFO] No se aplicó ningún filtro (¿faltó --all o algún flag?).")
    sys.exit(0)

# ==== GUARDAR (SOBRESCRIBE) ====
df.to_csv(csv_path, index=False)
print(f"[OK] CSV sobrescrito con columnas de filtros: {csv_path}")
