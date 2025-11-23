# RFCribadoDianasMoleculares
Un programa para entrenar modelos RF para el cribado de dianas moleculares

[![PyPI version](https://img.shields.io/pypi/v/numpy)](https://pypi.org/project/numpy) 

## Índice

- [Descripción](#descripcion)
- [Requisitos](#requisitos)
- [Inicio rapido](#inicio-rapido)
- [Casos de uso](#casos-de-uso)
- [Argumentos de scripts](#argumentos-de-scripts)
- [Limitaciones](#limitaciones)
- [Bibliografia](#bibliografia)

## Descripcion

Tienes la posibilidad de ejecutar un programa para archivos .csv que se encuentren en la carpeta Datasets/Brutos 
que realizara un procesamiento para generar los descriptores con los filtros elegidos, y posteriormente entrenara 
modelo de Random Forest (RF) para hacer un cribado de moleculas activas y no activas para la diana segun el I50.

## Requisitos

Necesitas tener instalado python y RKDirt para hacer funcionar el programa. A demas a sido creado en un sistema 
Windous, en principio no tiene que causar problemas en linux o mac.

## Casos de uso

En caso de querer un cribado rapido donde el modelo te diga si tiene actividad o no cada molecula, y tengas un set 
de datos de mas de 100 muestras para el entrenamiento.

## Inicio rapido:

Entrenamiento típico: python setup.py --all

Predicción típica: python setup.py --predict --all --notrain

## Argumentos de scripts

setup.py (raíz)

--predict → Ejecuta el pipeline en modo predicción (descriptor/filtros operan en Dataset/Prediccion/).

--notrain → No entrena; al final ejecuta predictor.py.

--all → Aplica todos los filtros en filtros.py.

--pains → Aplica solo filtro PAINS.

--chelators → Aplica solo filtro de quelación pan-metalo.

--druglikeness → Calcula violaciones Lipinski/Veber.

Passthrough a model.py: --dir --ic50 --train --val --model --retrain --algo --name --rf_n_estimators --rf_max_depth --rf_min_samples_leaf --rf_step --patience --xgb_n_estimators --xgb_max_depth --xgb_learning_rate --xgb_subsample --xgb_colsample_bytree --xgb_reg_lambda --xgb_early_stopping --xgb_verbose

Passthrough a predictor.py (si --notrain): --dir --model --threshold

Library/descriptor.py

--predict → Guarda en Dataset/Prediccion/descriptores.csv (por defecto guarda en Dataset/Procesados/).

Library/filtros.py

--dir <csv> → CSV de entrada/salida (por defecto Dataset/Procesados/descriptores.csv).

--predict → Usa Dataset/Prediccion/descriptores.csv como entrada/salida por defecto.

--all → Aplica PAINS + quelación + Lipinski/Veber.

--pains → Solo PAINS.

--chelators → Solo quelación pan-metalo.

--druglikeness → Añade Lipinski_violations / Veber_violations.

Library/model.py

--dir <csv> → CSV de descriptores (por defecto Dataset/Procesados/descriptores.csv).

--ic50 <0–1> → Percentil para etiquetar “activo” (por defecto 0.5 = 50% más potentes → 1).

--train <0–1> / --val <0–1> → Splits (por defecto 0.75 / 0.15; test = resto).

--model base|new|test|<nombre> → base: RF por defecto (usa/crea checkpoint “base”) · new: crea modelo con --algo y --name · test: búsqueda rápida.

--retrain → Fuerza re-entrenar aunque exista checkpoint.

--algo rf|xgb → Algoritmo (solo con --model new).

--name <cadena> → Nombre/carpeta del modelo (solo con --model new).

RF (si --algo rf o base):

--rf_n_estimators (def. 2500) · --rf_max_depth (def. None) · --rf_min_samples_leaf (def. 1)

--rf_step (def. 100; crecimiento por iteración) · --patience (def. 15; early stop en val)

XGB (si --algo xgb):

--xgb_n_estimators (def. 2000) · --xgb_max_depth (def. 6) · --xgb_learning_rate (def. 0.1)

--xgb_subsample (def. 1.0) · --xgb_colsample_bytree (def. 1.0) · --xgb_reg_lambda (def. 1.0)

--xgb_early_stopping (def. 50) · --xgb_verbose (log detallado)

Library/predictor.py

--model <nombre> → Carpeta en Models/<nombre>/checkpoints/ (defecto: base).

--dir <csv> → CSV de predicción (defecto Dataset/Prediccion/descriptores.csv).

--threshold <0–1> → Umbral para activo_pred (defecto 0.5).

## Limitaciones

El modelo entrena sin aplicar los filtros, estos son solo orientativos para el observador. Necesitas descargar los datasets
y entrenar un modelo diferente para cada conjunto de datasets.

## Bibliografia

Winer A, Adams S, Mignatti P. Matrix metalloproteinase inhibitors in 
cancer therapy: Turning past failures into future successes. Vol. 17, 
Molecular Cancer Therapeutics. 2018.  
27 
1. Fields GB. The Rebirth of Matrix Metalloproteinase Inhibitors: Moving 
Beyond the Dogma. Cells. 2019 Aug;8(9).  
2. Tropsha A, Isayev O, Varnek A, Schneider G, Cherkasov A. Integrating 
QSAR modelling and deep learning in drug discovery: the emergence of 
deep QSAR. Nat Rev Drug Discov [Internet]. 2024;23(2):141–55. 
Available from: https://doi.org/10.1038/s41573-023-00832-0 
3. Warren GL, Andrews CW, Capelli AM, Clarke B, LaLonde J, Lambert MH, 
et al. A critical assessment of docking programs and scoring functions. J 
Med Chem. 2006 Oct;49(20):5912–31.  
4. Cheng T, Li Q, Zhou Z, Wang Y, Bryant SH. Structure-based virtual 
screening for drug discovery: a problem-centric review. AAPS J. 2012 
Mar;14(1):133–41.  
5. McNutt AT, Francoeur P, Aggarwal R, Masuda T, Meli R, Ragoza M, et al. 
GNINA 1.0: molecular docking with deep learning. J Cheminform. 
2021;13(1).  
6. Li P, Merz KMJ. Metal Ion Modeling Using Classical Mechanics. Chem 
Rev. 2017 Feb;117(3):1564–686.  
7. Svetnik V, Liaw A, Tong C, Culberson JC, Sheridan RP, Feuston BP. 
Random forest: a classification and regression tool for compound 
classification  and QSAR modeling. J Chem Inf Comput Sci. 
2003;43(6):1947–58.  
8. Wu Z, Zhu M, Kang Y, Leung ELH, Lei T, Shen C, et al. Do we need 
different machine learning algorithms for QSAR modeling? A 
comprehensive assessment of 16 machine learning algorithms on 14 
QSAR data sets. Brief Bioinform [Internet]. 2020;22(4):bbaa321. Available 
from: https://doi.org/10.1093/bib/bbaa321 



