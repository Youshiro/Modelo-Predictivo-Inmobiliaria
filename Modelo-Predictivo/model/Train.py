# Importamos las librerías necesarias para entrenar y persistir nuestro modelo
import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# El método train_model se encargará de entrenar el modelo cuando la API lo necesite
def train_model():
    
    # Cargamos el conjunto de datos y creamos un dataframe de pandas
    datos = pd.read_csv('./model/mlinmo.csv')

    # Asignamos los valores de las columnas a variables independientes o predictoras
    years = datos['years'].values
    metros = datos['metros'].values
    habitaciones = datos['habitaciones'].values
    servicios = datos['servicios'].values
    garaje = datos['garaje'].values
    ascensor = datos['ascensor'].values
    trastero = datos['trastero'].values
    tipo = datos['tipo'].values
    precio = datos['precio'].values
    
    # Creamos un array X con las variables independientes
    X = np.array([years, 
              metros, 
              habitaciones, 
              servicios, 
              garaje, 
              ascensor, 
              trastero, 
              tipo]).T
    
    # Creamos un array Y con la variable dependiente
    Y = np.array(precio)

    # Dividimos el dataset en un 20% de datos de test y un 80% en datos de entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    # Creamos un objeto regresion utilizando el modelo LinearRegression de la librería sckit-learn
    regresion = LinearRegression()
    
    # Entrenamos el modelo y realizamos el predict con los datos de test
    regresion = regresion.fit(X_train, y_train)
    Y_pred = regresion.predict(X_test)

    # Calculamos el error y la precisión del modelo
    error = np.sqrt(mean_squared_error(y_test, Y_pred))
    r2 = regresion.score(X_train, y_train)

    # Persistimos el modelo con la utilidad joblib
    joblib.dump(regresion, 'mlinmo.model')

    # print("El error es: ", error)
    print("El valor de r2 es: ", r2)
    # print("Los coeficientes son: \n", regresion.coef_)
    