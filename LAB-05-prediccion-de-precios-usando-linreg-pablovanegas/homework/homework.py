#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
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
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import json
import gzip
import pickle
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def load_and_clean_data():
    """ 
    PASO 1: PREPROCESANDO DE LOS DATOS
    
    return: TRAIN and TEST dataframes cleaned

    """

    train_data = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_data = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

# Crear la columna 'Age' a partir de la columna 'Year'
    train_data['Age'] = 2021 - train_data['Year']
    test_data['Age'] = 2021 - test_data['Year']

# Eliminar las columnas 'Year' y 'Car_Name'
    train_data = train_data.drop(columns=['Year', 'Car_Name'])
    test_data = test_data.drop(columns=['Year', 'Car_Name'])

    return train_data, test_data


def split_data(train_data, test_data):
    """ 
    PASO 2: DIVIDIENDO LOS DATOS EN x E Y
    
    return: x_train, y_train, x_test, y_test

    """

    x_train = train_data.drop(columns=['Selling_Price'])
    y_train = train_data['Selling_Price']
    x_test = test_data.drop(columns=['Selling_Price'])
    y_test = test_data['Selling_Price']

    return x_train, y_train, x_test, y_test


#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#


def create_pipeline(x_train):
    """
    paso 3: CREANDO EL PIPELINE

    """
    # Identificando columnas categoricas y numericas
    categprocal_cols = x_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = ['Driven_kms', 'Age', 'Owner']  # <-- corrige aquí
    categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']  # <-- corrige aquí

    # Creando el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    # Creando el pipeline
    pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor), # Preprocesador
        # k mejores entradas
        ('selector', SelectKBest()),
        ('regressor', LinearRegression())  # Modelo de regresion lineal
    ]
    )

    return pipeline


# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#

def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    paso 4: OPTIMIZANDO HIPERPARAMETROS
    """
    param_grid = {
        'selector__k': [5, 7, 9, 11],
        'preprocessor__cat__drop': ['first', None],  # Try different drop strategies for OneHotEncoder
        'preprocessor__num__feature_range': [(0, 1), (-1, 1)],  # Try different scaling ranges
        'regressor__fit_intercept': [True, False]  # Whether to calculate the intercept
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(x_train, y_train)
    return grid_search
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#

def save_model(model):
    """
    paso 5: GUARDANDO EL MODELO
    """

    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)

#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test):
    """
    paso 6: CALCULANDO Y GUARDANDO LAS METRICAS
    """

    os.makedirs("files/output", exist_ok=True)
    metrics = []

    # Predicciones para el conjunto de entrenamiento
    y_train_pred = model.predict(x_train)
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mad_train = mean_absolute_error(y_train, y_train_pred)

    metrics.append({
        'type': 'metrics',
        'dataset': 'train',
        'r2': r2_train,
        'mse': mse_train,
        'mad': mad_train
    })

    # Predicciones para el conjunto de prueba
    y_test_pred = model.predict(x_test)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mad_test = mean_absolute_error(y_test, y_test_pred)

    metrics.append({
        'type': 'metrics',
        'dataset': 'test',
        'r2': r2_test,
        'mse': mse_test,
        'mad': mad_test
    })

    # Guardando las metricas en un archivo JSON
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")


# MAIN  
def main():
    # Paso 1
    train_data, test_data = load_and_clean_data()

    # Paso 2
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)
    # imprimiento las columnas de x_train para verificar
    print("Columnas de x_train:", x_train.columns.tolist())

    # Paso 3
    pipeline = create_pipeline(x_train)

    # Paso 4
    optimized_model = optimize_hyperparameters(pipeline, x_train, y_train)
    print("Best hyperparameters:", optimized_model.best_params_)

    # Paso 5
    # Save the GridSearchCV object instead of best_estimator_
    save_model(optimized_model)

    # paso 6
    # Use the full GridSearchCV model for metrics
    calculate_and_save_metrics(optimized_model, x_train, y_train, x_test, y_test)

    print('Proceso completado.')
    print(f"Mejor score en CV: {-optimized_model.best_score_}")


if __name__ == "__main__":
    main()