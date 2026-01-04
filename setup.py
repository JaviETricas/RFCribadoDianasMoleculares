#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
import subprocess
from pathlib import Path

# ======================
# Rutas relativas
# ======================
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent
LIB_DIR   = REPO_ROOT / "Library"

DESC_PY = LIB_DIR / "descriptor.py"
FILT_PY = LIB_DIR / "filtros.py"
MODEL_PY = LIB_DIR / "model.py"
PRED_PY = LIB_DIR / "predictor.py"

# ======================
# Utilidades
# ======================
def rdkit_ok() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        import rdkit  # noqa: F401
        return True
    except Exception:
        return False

def run(cmd: list[str]) -> int:
    print(f"[CMD] {' '.join(map(str, cmd))}")
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        print(f"[ERROR] Falló el comando: {' '.join(map(str, cmd))} (code={completed.returncode})")
    return completed.returncode

def parse_known():
    import argparse
    p = argparse.ArgumentParser(description="Orquestador del pipeline (descriptor → filtros → model/predictor).",
                                add_help=True)
    # Control de flujo
    p.add_argument("--predict", action="store_true",
                   help="Hacer pipeline de PREDICCIÓN: descriptor/filtros trabajan sobre Dataset/Prediccion/.")
    p.add_argument("--notrain", action="store_true",
                   help="No entrenar; en su lugar ejecutar predictor.py al final.")
    p.add_argument("--descriptor_script", default=None,
                   help="Ruta a un descriptor alternativo (por ejemplo: Library/descriptor_sin_nm_v2_relajado.py).\n"
                        "Si no se indica, se usa Library/descriptor.py.")
    # Filtros
    p.add_argument("--all", action="store_true", help="Aplicar todos los filtros.")
    p.add_argument("--pains", action="store_true", help="Aplicar filtro PAINS.")
    p.add_argument("--chelators", action="store_true", help="Aplicar filtro de quelación pan-metalo.")
    p.add_argument("--druglikeness", action="store_true", help="Añadir nº de violaciones Lipinski/Veber.")
    # Todo lo demás lo dejamos como 'desconocido' para reenviar a model/predictor
    args, unknown = p.parse_known_args()
    return args, unknown

# Conjuntos de flags permitidas para reenvío
MODEL_FLAGS_VALUE = {
    "--dir","--ic50","--train","--val","--model","--algo","--name",
    "--rf_n_estimators","--rf_max_depth","--rf_min_samples_leaf","--rf_step","--patience",
    "--xgb_n_estimators","--xgb_max_depth","--xgb_learning_rate","--xgb_subsample",
    "--xgb_colsample_bytree","--xgb_reg_lambda","--xgb_early_stopping",
}
MODEL_FLAGS_BOOL = {"--retrain","--xgb_verbose"}

PRED_FLAGS_VALUE = {"--dir","--model","--threshold"}
PRED_FLAGS_BOOL  = set()  # de momento ninguno

def split_unknowns(unknown: list[str], allowed_value: set[str], allowed_bool: set[str]) -> list[str]:
    """Extrae de unknown solo las flags soportadas, conservando valores cuando proceda."""
    out = []
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok in allowed_bool:
            out.append(tok)
            i += 1
        elif tok in allowed_value:
            out.append(tok)
            if i + 1 < len(unknown):
                nxt = unknown[i+1]
                if not nxt.startswith("-"):
                    out.append(nxt)
                    i += 2
                else:
                    # flag con valor pero sin valor explícito → lo ignoramos con aviso
                    print(f"[AVISO] La flag {tok} esperaba un valor; se ignora por no estar presente.")
                    i += 1
            else:
                print(f"[AVISO] La flag {tok} esperaba un valor; se ignora por no estar presente.")
                i += 1
        else:
            # Flag no reconocida para este destino → ignorar en silencio
            i += 1
    return out

def main():
    # 0) RDKit presente
    if not rdkit_ok():
        print("[ERROR] No se pudo importar RDKit.")
        print("Instálalo en Windows (PowerShell) dentro de tu venv:")
        print("  python -m venv .venv")
        print("  .venv\\Scripts\\Activate.ps1")
        print("  pip install rdkit pandas numpy")
        return 1

    # 1) Argumentos
    args, unknown = parse_known()

    # 2) descriptor.py (o descriptor alternativo)
    desc_py = DESC_PY
    if args.descriptor_script:
        alt = Path(args.descriptor_script)
        # Resolver relativo al repo si procede
        alt = (REPO_ROOT / alt).resolve() if not alt.is_absolute() else alt.resolve()
        if not alt.exists():
            print(f"[ERROR] descriptor_script no existe: {alt}")
            return 1
        desc_py = alt

    desc_cmd = [sys.executable, str(desc_py)]
    if args.predict:
        desc_cmd.append("--predict")
    rc = run(desc_cmd)
    if rc != 0:
        return rc
    print("[OK] Datos brutos procesados. (descriptor.py)")

    # 3) filtros.py
    filt_cmd = [sys.executable, str(FILT_PY)]
    if args.predict:
        filt_cmd.append("--predict")
    # Añadir flags de filtros si se pidieron (si no, filtros.py saldrá con aviso y returncode=0)
    if args.all:          filt_cmd.append("--all")
    if args.pains:        filt_cmd.append("--pains")
    if args.chelators:    filt_cmd.append("--chelators")
    if args.druglikeness: filt_cmd.append("--druglikeness")

    rc = run(filt_cmd)
    if rc != 0:
        return rc
    print("[OK] Filtros aplicados. (filtros.py)")

    # 4) model o predictor
    if not args.notrain:
        # Entrenamiento
        model_cmd = [sys.executable, str(MODEL_PY)]
        model_forward = split_unknowns(unknown, MODEL_FLAGS_VALUE, MODEL_FLAGS_BOOL)
        model_cmd.extend(model_forward)
        rc = run(model_cmd)
        if rc != 0:
            return rc
        print("[OK] Modelo entrenado y guardado. (model.py)")
    else:
        # Predicción
        pred_cmd = [sys.executable, str(PRED_PY)]
        pred_forward = split_unknowns(unknown, PRED_FLAGS_VALUE, PRED_FLAGS_BOOL)
        pred_cmd.extend(pred_forward)
        rc = run(pred_cmd)
        if rc != 0:
            return rc
        print("[OK] Resultados generados. Consulta la carpeta 'Resultados/'. (predictor.py)")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
