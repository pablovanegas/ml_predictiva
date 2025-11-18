# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

from __future__ import annotations

import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore
from sklearn.metrics import (
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore


DATA_DIR = Path("files/input")
MODEL_PATH = Path("files/models/model.pkl.gz")
OUTPUT_METRICS = Path("files/output/metrics.json")


def _ensure_dirs() -> None:
	MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Load train and test CSV (zipped) as DataFrames.

	Returns:
		Tuple of (train_df, test_df)
	"""
	train_fp = DATA_DIR / "train_data.csv.zip"
	test_fp = DATA_DIR / "test_data.csv.zip"
	train_df = pd.read_csv(train_fp, compression="zip")
	test_df = pd.read_csv(test_fp, compression="zip")
	return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Apply dataset cleaning rules.

	- Rename 'default payment next month' to 'default'
	- Drop 'ID'
	- Drop rows with missing values
	- Cap EDUCATION > 4 to 4 (others)
	"""
	df = df.copy()

	# Standardize target column name
	if "default payment next month" in df.columns:
		df = df.rename(columns={"default payment next month": "default"})

	# Drop ID column if present 
	if "ID" in df.columns:
		df = df.drop(columns=["ID"])  # type: ignore[arg-type]

	# Normalize EDUCATION values > 4 to 4 (others)
	if "EDUCATION" in df.columns:
		df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

	# Drop rows with NAs
	df = df.dropna(axis=0).reset_index(drop=True)

	return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
	"""Split features and target.

	Returns:
		X (DataFrame), y (Series)
	"""
	target_col = "default"
	assert target_col in df.columns, "Target column 'default' not found after cleaning."
	X = df.drop(columns=[target_col])
	y = df[target_col].astype(int)
	return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
	"""Build the classification pipeline with the required steps.

	Steps:
	  - OneHotEncoder for categorical columns
	  - PCA using all components
	  - StandardScaler
	  - SelectKBest
	  - SVC
	"""
	# Heuristically define categorical columns
	cat_cols: List[str] = [
		"SEX",
		"EDUCATION",
		"MARRIAGE",
		"PAY_0",
		"PAY_2",
		"PAY_3",
		"PAY_4",
		"PAY_5",
		"PAY_6",
	]
	cat_cols = [c for c in cat_cols if c in X.columns]
	num_cols = [c for c in X.columns if c not in cat_cols]

	# Force dense output from OHE so PCA can consume it reliably across sklearn versions
	ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	preprocessor = ColumnTransformer(
		transformers=[
			("cat", ohe, cat_cols),
			("num", "passthrough", num_cols),
		]
	)

	pipe = Pipeline(
		steps=[
			("preprocess", preprocessor),
			("pca", PCA(n_components=None, random_state=42)),
			("scaler", StandardScaler()),
			("select", SelectKBest(score_func=f_classif, k=50)),
			("svc", SVC(kernel="rbf", class_weight="balanced", random_state=42)),
		]
	)
	return pipe


def tune_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
	"""Tune hyperparameters with 10-fold cross-validation using balanced accuracy."""
	# Reasonable, compact grid to keep runtime acceptable while meeting thresholds
	param_grid = {
		"select__k": [20, 40],
		"svc__C": [ 1.0, 2.0],
		"svc__gamma": ["scale", "auto"],
		"svc__kernel": ["rbf"],
	}

	cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	grid = GridSearchCV(
		estimator=pipe,
		param_grid=param_grid,
		scoring="balanced_accuracy",
		cv=cv,
		n_jobs=-1,
		refit=True,
		verbose=0,
	)
	grid.fit(X, y)
	return grid


def save_model(model: GridSearchCV, path: Path = MODEL_PATH) -> None:
	"""Save model compressed with gzip pickle."""
	with gzip.open(path, "wb") as f:
		pickle.dump(model, f)


def _compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, dataset: str) -> Dict[str, object]:
	return {
		"type": "metrics",
		"dataset": dataset,
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
	}


def _compute_cm_dict(y_true: np.ndarray, y_pred: np.ndarray, dataset: str) -> Dict[str, object]:
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
	# cm layout with labels [0,1]: rows=true, cols=pred
	t0_p0 = int(cm[0, 0])
	t0_p1 = int(cm[0, 1])
	t1_p0 = int(cm[1, 0])
	t1_p1 = int(cm[1, 1])
	return {
		"type": "cm_matrix",
		"dataset": dataset,
		"true_0": {"predicted_0": t0_p0, "predicted_1": t0_p1},
		"true_1": {"predicted_0": t1_p0, "predicted_1": t1_p1},
	}


def evaluate_and_save_metrics(model: GridSearchCV, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, path: Path = OUTPUT_METRICS) -> None:
	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)

	y_train_np = np.asarray(y_train)
	y_test_np = np.asarray(y_test)
	y_pred_train_np = np.asarray(y_pred_train)
	y_pred_test_np = np.asarray(y_pred_test)

	# Build entries in the required order: metrics train, metrics test, cm train, cm test
	entries: List[Dict[str, object]] = [
		_compute_metrics_dict(y_train_np, y_pred_train_np, "train"),
		_compute_metrics_dict(y_test_np, y_pred_test_np, "test"),
		_compute_cm_dict(y_train_np, y_pred_train_np, "train"),
		_compute_cm_dict(y_test_np, y_pred_test_np, "test"),
	]

	with open(path, "w", encoding="utf-8") as f:
		for row in entries:
			f.write(json.dumps(row) + "\n")


def main() -> None:
	_ensure_dirs()

	# Load and clean
	train_df, test_df = load_data()
	train_df = clean_data(train_df)
	test_df = clean_data(test_df)

	# Split
	X_train, y_train = split_xy(train_df)
	X_test, y_test = split_xy(test_df)

	# Build, tune, save
	pipe = build_pipeline(X_train)
	model = tune_model(pipe, X_train, y_train)
	save_model(model, MODEL_PATH)

	# Evaluate and save metrics
	evaluate_and_save_metrics(model, X_train, y_train, X_test, y_test, OUTPUT_METRICS)


if __name__ == "__main__":
	main()

