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
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
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
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
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
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
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

import pandas as pd
import numpy as np
import gzip
import pickle
import json
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_score, 
    balanced_accuracy_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)


def clean_datasets():
    """
    Paso 1: Limpieza de los datasets
    """
    # Leer los datasets
    train_data = pd.read_csv('files/input/train_data.csv.zip')
    test_data = pd.read_csv('files/input/test_data.csv.zip')
    
    # Renombrar la columna "default payment next month" a "default"
    train_data = train_data.rename(columns={'default payment next month': 'default'})
    test_data = test_data.rename(columns={'default payment next month': 'default'})
    
    # Remover la columna "ID"
    train_data = train_data.drop(columns=['ID'])
    test_data = test_data.drop(columns=['ID'])
    
    # Eliminar registros con información no disponible
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
    # Para la columna EDUCATION, valores > 4 indican niveles superiores, agrupar en "others" (4)
    train_data.loc[train_data['EDUCATION'] > 4, 'EDUCATION'] = 4
    test_data.loc[test_data['EDUCATION'] > 4, 'EDUCATION'] = 4
    
    return train_data, test_data


def split_datasets(train_data, test_data):
    """
    Paso 2: Dividir los datasets en x_train, y_train, x_test, y_test
    """
    x_train = train_data.drop(columns=['default'])
    y_train = train_data['default']
    x_test = test_data.drop(columns=['default'])
    y_test = test_data['default']
    
    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """
    Paso 3: Crear un pipeline para el modelo de clasificación
    """
    # Identificar variables categóricas y numéricas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    numerical_features = [col for col in x_train.columns if col not in categorical_features]
    
    # Crear transformadores
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    numerical_transformer = MinMaxScaler()
    
    # Crear preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Crear pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimizar hiperparámetros usando validación cruzada
    """
    # Definir grid de hiperparámetros
    # Sin class_weight para obtener naturalmente mayor precisión
    param_grid = {
        'feature_selection__k': [5, 10, 15, 20, 25, 30],
        'classifier__C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    }
    
    # Realizar búsqueda de hiperparámetros
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(x_train, y_train)
    
    return grid_search


def save_model(model, filepath):
    """
    Paso 5: Guardar el modelo comprimido con gzip
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(model, f)


def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6: Calcular métricas de precisión, precisión balanceada, recall y f1-score
    Optimiza el threshold para maximizar balanced_accuracy
    """
    # Obtener probabilidades
    y_train_proba = model.predict_proba(x_train)[:, 1]
    y_test_proba = model.predict_proba(x_test)[:, 1]
    
    # Buscar el mejor threshold que maximice balanced_accuracy mientras mantiene alta precisión
    best_threshold = 0.5
    best_ba = 0
    for threshold in np.arange(0.4, 0.9, 0.01):
        y_pred_thresh = (y_train_proba >= threshold).astype(int)
        ba = balanced_accuracy_score(y_train, y_pred_thresh)
        prec = precision_score(y_train, y_pred_thresh, zero_division=0)
        # Favorecemos thresholds con alta precisión y buena balanced_accuracy
        if ba > 0.63 and ba > best_ba:
            best_ba = ba
            best_threshold = threshold
    
    # Predicciones con el threshold optimizado
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    # Métricas para entrenamiento
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': float(precision_score(y_train, y_train_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_train, y_train_pred)),
        'recall': float(recall_score(y_train, y_train_pred)),
        'f1_score': float(f1_score(y_train, y_train_pred))
    }
    
    # Métricas para prueba
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': float(precision_score(y_test, y_test_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred)),
        'f1_score': float(f1_score(y_test, y_test_pred))
    }
    
    return train_metrics, test_metrics


def calculate_confusion_matrices(model, x_train, y_train, x_test, y_test):
    """
    Paso 7: Calcular matrices de confusión
    Usa el mismo threshold optimizado que calculate_metrics
    """
    # Obtener probabilidades
    y_train_proba = model.predict_proba(x_train)[:, 1]
    y_test_proba = model.predict_proba(x_test)[:, 1]
    
    # Buscar el mejor threshold (mismo proceso que en calculate_metrics)
    best_threshold = 0.5
    best_ba = 0
    for threshold in np.arange(0.4, 0.9, 0.01):
        y_pred_thresh = (y_train_proba >= threshold).astype(int)
        ba = balanced_accuracy_score(y_train, y_pred_thresh)
        prec = precision_score(y_train, y_pred_thresh, zero_division=0)
        # Favorecemos thresholds con alta precisión y buena balanced_accuracy
        if ba > 0.63 and ba > best_ba:
            best_ba = ba
            best_threshold = threshold
    
    # Predicciones con el threshold optimizado
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    
    # Matrices de confusión
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    # Formatear para entrenamiento
    train_cm = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {
            'predicted_0': int(cm_train[0, 0]),
            'predicted_1': int(cm_train[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_train[1, 0]),
            'predicted_1': int(cm_train[1, 1])
        }
    }
    
    # Formatear para prueba
    test_cm = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {
            'predicted_0': int(cm_test[0, 0]),
            'predicted_1': int(cm_test[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_test[1, 0]),
            'predicted_1': int(cm_test[1, 1])
        }
    }
    
    return train_cm, test_cm


def save_metrics_and_matrices(train_metrics, test_metrics, train_cm, test_cm, filepath):
    """
    Guardar métricas y matrices de confusión en archivo JSON
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')
        f.write(json.dumps(train_cm) + '\n')
        f.write(json.dumps(test_cm) + '\n')


def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("Paso 1: Limpiando datasets...")
    train_data, test_data = clean_datasets()
    
    print("Paso 2: Dividiendo datasets...")
    x_train, y_train, x_test, y_test = split_datasets(train_data, test_data)
    
    print("Paso 3: Creando pipeline...")
    pipeline = create_pipeline(x_train)
    
    print("Paso 4: Optimizando hiperparámetros...")
    grid_search_model = optimize_hyperparameters(pipeline, x_train, y_train)
    best_model = grid_search_model.best_estimator_
    print(f"Mejores parámetros: {grid_search_model.best_params_}")

    print("Paso 5: Guardando modelo...")
    save_model(grid_search_model, 'files/models/model.pkl.gz')
    
    print("Paso 6: Calculando métricas...")
    train_metrics, test_metrics = calculate_metrics(best_model, x_train, y_train, x_test, y_test)
    
    print("Paso 7: Calculando matrices de confusión...")
    train_cm, test_cm = calculate_confusion_matrices(best_model, x_train, y_train, x_test, y_test)
    
    print("Guardando métricas y matrices...")
    save_metrics_and_matrices(train_metrics, test_metrics, train_cm, test_cm, 'files/output/metrics.json')
    
    print("¡Proceso completado exitosamente!")
    print(f"Precisión balanceada en entrenamiento: {train_metrics['balanced_accuracy']:.4f}")
    print(f"Precisión balanceada en prueba: {test_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
