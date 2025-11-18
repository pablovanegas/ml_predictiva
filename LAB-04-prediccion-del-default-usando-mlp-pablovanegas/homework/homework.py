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
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
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

import os
import gzip
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_and_clean_data():
    """
    Paso 1: Realiza la limpieza de los datasets.
    """
    # Cargar datos
    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    # Renombrar columna "default payment next month" a "default"
    train_data = train_data.rename(columns={"default payment next month": "default"})
    test_data = test_data.rename(columns={"default payment next month": "default"})

    # Remover columna "ID"
    if "ID" in train_data.columns:
        train_data = train_data.drop(columns=["ID"])
    if "ID" in test_data.columns:
        test_data = test_data.drop(columns=["ID"])

    # Eliminar registros con información no disponible (valores 0 en EDUCATION y MARRIAGE)
    train_data = train_data[
        (train_data["EDUCATION"] != 0) & (train_data["MARRIAGE"] != 0)
    ]
    test_data = test_data[(test_data["EDUCATION"] != 0) & (test_data["MARRIAGE"] != 0)]

    # Para la columna EDUCATION, valores > 4 indican niveles superiores, agrupar en categoría 4 (others)
    train_data.loc[train_data["EDUCATION"] > 4, "EDUCATION"] = 4
    test_data.loc[test_data["EDUCATION"] > 4, "EDUCATION"] = 4

    return train_data, test_data


def split_data(train_data, test_data):
    """
    Paso 2: Divide los datasets en x_train, y_train, x_test, y_test.
    """
    x_train = train_data.drop(columns=["default"])
    y_train = train_data["default"]

    x_test = test_data.drop(columns=["default"])
    y_test = test_data["default"]

    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """
    Paso 3: Crea un pipeline para el modelo de clasificación.
    """
    # Identificar columnas categóricas y numéricas
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_cols = [col for col in x_train.columns if col not in categorical_cols]

    # Transformador de columnas para one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    # Pipeline completo
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", PCA()),  # Usa todas las componentes
            ("scaler", StandardScaler()),  # Escala los datos
            ("selector", SelectKBest(score_func=f_classif)),
            ("classifier", MLPClassifier(max_iter=1000, random_state=42)),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimiza los hiperparámetros del pipeline usando validación cruzada.
    """
    # Grid de hiperparámetros
    param_grid = {
        "pca__n_components": [None, 10],
        "selector__k": [10, "all"],
        "classifier__hidden_layer_sizes": [(50,), (100,)],
        "classifier__activation": ["relu"],
        "classifier__solver": ["adam"],
        "classifier__alpha": [0.001, 0.01],
        "classifier__learning_rate": ["constant"],
    }

    # GridSearchCV con validación cruzada de 10 splits
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(x_train, y_train)

    return grid_search

def save_model(model):
    """
    Paso 5: Guarda el modelo comprimido con gzip.
    """
    os.makedirs("files/models", exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6 y 7: Calcula y guarda las métricas y matrices de confusión.
    """
    os.makedirs("files/output", exist_ok=True)

    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calcular métricas para entrenamiento
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": float(precision_score(y_train, y_train_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
        "f1_score": float(f1_score(y_train, y_train_pred)),
    }

    # Calcular métricas para prueba
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": float(precision_score(y_test, y_test_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "recall": float(recall_score(y_test, y_test_pred)),
        "f1_score": float(f1_score(y_test, y_test_pred)),
    }

    # Matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    train_cm = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }

    test_cm = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }

    # Guardar en archivo
    with open("files/output/metrics.json", "w") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")
        f.write(json.dumps(train_cm) + "\n")
        f.write(json.dumps(test_cm) + "\n")


def main():
    """
    Función principal que ejecuta todos los pasos.
    """
    # Paso 1: Limpiar datos
    train_data, test_data = load_and_clean_data()

    # Paso 2: Dividir datos
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)

    # Paso 3: Crear pipeline
    pipeline = create_pipeline(x_train)

    # Paso 4: Optimizar hiperparámetros
    grid_search_model = optimize_hyperparameters(pipeline, x_train, y_train)
    best_model = grid_search_model.best_estimator_
    print(f"Mejores parámetros: {grid_search_model.best_params_}")
   
    # Paso 5: Guardar modelo
    save_model(grid_search_model)

    # Paso 6 y 7: Calcular y guardar métricas
    calculate_and_save_metrics(best_model, x_train, y_train, x_test, y_test)

    print("Modelo entrenado y guardado exitosamente!")
    print(f"Mejor score de validación cruzada: {best_model.score(x_train, y_train)}")


if __name__ == "__main__":
    main()
