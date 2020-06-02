# Importamos las librerías necesarias y los métodos

import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model.Train import train_model
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)

# Si el modelo no existe, realizamos una llamada a train_model() para que sea reentrenado
if not os.path.isfile('mlinmo.model'):
    train_model()


# Cargamos el modelo con la utilidad joblib
model = joblib.load('mlinmo.model')


# Realizamos la predicción, convertimos los datos a float, que es el tipo de dato que necesita el algoritmo y llamamos al método predict del modelo
# enviándole los datos recogidos
class MakePrediction(Resource):
    @staticmethod
    def post():

        posted_data = request.get_json()
        years = float(posted_data['years'])
        metros = float(posted_data['metros'])
        habitaciones = float(posted_data['habitaciones'])
        servicios = float(posted_data['servicios'])
        garaje = float(posted_data['garaje'])
        ascensor = float(posted_data['ascensor'])
        trastero = float(posted_data['trastero'])
        tipo = float(posted_data['tipo'])

        prediction = model.predict([[years, metros, habitaciones, servicios, garaje, ascensor, trastero, tipo]]).tolist()
        # Obtenemos un resultado que convertimos a JSON
        return jsonify({
            'Prediction': float('%.2f' % prediction[0])
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)