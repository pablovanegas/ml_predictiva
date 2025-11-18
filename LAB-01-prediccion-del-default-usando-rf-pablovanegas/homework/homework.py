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
# - Ajusta un modelo de bosques aleatorios (rando forest).
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

import pandas as pd
import numpy as np
import gzip
import pickle
import json
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, balanced_accuracy_score, recall_score, 
    f1_score, confusion_matrix
)
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

def load_and_clean_data():
    """
    Paso 1: Carga y limpieza de los datasets
    """
    # Cargar los datos
    train_df = pd.read_csv('files/input/train_default_of_credit_card_clients.csv')
    test_df = pd.read_csv('files/input/test_default_of_credit_card_clients.csv')
    
    def clean_dataset(df):
        # Renombrar la columna objetivo
        df = df.rename(columns={'default payment next month': 'default'})
        
        # Remover la columna ID
        df = df.drop('ID', axis=1)
        
        # Eliminar registros con información no disponible (valores NaN)
        df = df.dropna()
        
        # Para la columna EDUCATION, agrupar valores > 4 en la categoría "others" (4)
        df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
        
        return df
    
    train_cleaned = clean_dataset(train_df)
    test_cleaned = clean_dataset(test_df)
    
    return train_cleaned, test_cleaned

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

def split_features_target(train_df, test_df):
    """
    Paso 2: Dividir los datasets en X y y
    """
    # Separar características y variable objetivo
    x_train = train_df.drop('default', axis=1)
    y_train = train_df['default']
    x_test = test_df.drop('default', axis=1)
    y_test = test_df['default']
    
    return x_train, y_train, x_test, y_test

#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
# En homework.py

from sklearn.preprocessing import StandardScaler

def create_pipeline():
    """
    Paso 3: Crear pipeline con preprocesamiento y Random Forest
    """
    # Identificar variables categóricas y numéricas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 
                            'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # Todas las demás columnas se asumirán como numéricas
    
    # Crear transformador para variables categóricas
    categorical_transformer = OneHotEncoder(drop='first', 
                                            sparse_output=False, 
                                            handle_unknown='ignore')
    
    # Crear transformador para variables numéricas
    numeric_transformer = StandardScaler()

    # Crear el preprocesador que aplica diferentes transformaciones a diferentes columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Dejar las columnas restantes como están (si las hubiera)
    )
    
    # Crear el pipeline final
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
# En la función optimize_hyperparameters
# En homework.py
# En homework.py, reemplaza esta función completa:

def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimización de hiperparámetros con validación cruzada
    """
    # Grilla de búsqueda balanceada para evitar el sobreajuste
    param_grid = {
        'classifier__n_estimators': [200,300],
        'classifier__max_depth': [10,15,None],
        'classifier__min_samples_split': [5,10],
        'classifier__min_samples_leaf': [4],
        'classifier__class_weight': ['balanced_subsample'],
        'classifier__max_features': ['sqrt']
    }
    
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

# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.


def save_model(model, filepath):
    """
    Paso 5: Guardar el modelo comprimido
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with gzip.open(filepath, 'wb') as f:  # type: ignore
        pickle.dump(model, f)

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

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6: Calcular métricas para entrenamiento y prueba
    """
    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Métricas para entrenamiento
    train_metrics = {
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1_score': f1_score(y_train, y_train_pred)
    }
    
    # Métricas para prueba
    test_metrics = {
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred)
    }
    
    return train_metrics, test_metrics, y_train_pred, y_test_pred


# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

def calculate_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred):
    """
    Paso 7: Calcular matrices de confusión
    """
    # Matriz de confusión para entrenamiento
    cm_train = confusion_matrix(y_train, y_train_pred)
    train_cm_dict = {
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
    
    # Matriz de confusión para prueba
    cm_test = confusion_matrix(y_test, y_test_pred)
    test_cm_dict = {
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
    
    return train_cm_dict, test_cm_dict


def save_metrics_to_json(train_metrics, test_metrics, train_cm_dict, test_cm_dict, filepath):
    """
    Guardar métricas y matrices de confusión en archivo JSON
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        # Escribir métricas
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')
        # Escribir matrices de confusión
        f.write(json.dumps(train_cm_dict) + '\n')
        f.write(json.dumps(test_cm_dict) + '\n')


def main():
    """
    Función principal que ejecuta todo el pipeline
    """
    print("Iniciando proceso de machine learning...")
    
    # Paso 1: Cargar y limpiar datos
    print("Paso 1: Cargando y limpiando datos...")
    train_df, test_df = load_and_clean_data()
    print(f"Datos de entrenamiento: {train_df.shape}")
    print(f"Datos de prueba: {test_df.shape}")
    
    # Paso 2: Dividir características y variable objetivo
    print("Paso 2: Dividiendo características y variable objetivo...")
    x_train, y_train, x_test, y_test = split_features_target(train_df, test_df)
    
    # Paso 3: Crear pipeline
    print("Paso 3: Creando pipeline...")
    pipeline = create_pipeline()
    
    # Paso 4: Optimizar hiperparámetros
    print("Paso 4: Optimizando hiperparámetros...")
    grid_search_model = optimize_hyperparameters(pipeline, x_train, y_train)
    best_model = grid_search_model.best_estimator_
    print(f"Mejores parámetros: {grid_search_model.best_params_}")

    # Paso 5: Guardar modelo
    print("Paso 5: Guardando modelo...")
    save_model(grid_search_model, 'files/models/model.pkl.gz')
    
    # Paso 6: Calcular métricas
    print("Paso 6: Calculando métricas...")
    train_metrics, test_metrics, y_train_pred, y_test_pred = calculate_metrics(
        best_model, x_train, y_train, x_test, y_test
    )
    
    # Paso 7: Calcular matrices de confusión
    print("Paso 7: Calculando matrices de confusión...")
    train_cm_dict, test_cm_dict = calculate_confusion_matrices(
        y_train, y_train_pred, y_test, y_test_pred
    )
    
    # Guardar métricas y matrices de confusión
    save_metrics_to_json(
        train_metrics, test_metrics, train_cm_dict, test_cm_dict,
        'files/output/metrics.json'
    )
    
    print("Proceso completado exitosamente!")
    print(f"Métricas de entrenamiento: {train_metrics}")
    print(f"Métricas de prueba: {test_metrics}")


if __name__ == "__main__":
    main()
