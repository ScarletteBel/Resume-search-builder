from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predecir", methods=["POST"])
def predecir():
    cuartos=int(request.form["rooms"])
    distancia=int(request.form["distance"])
    prediccion= model.predict([[cuartos, distancia]])
    output=round(prediccion[0], 2)
    return render_template('index.html', prediccion_texto=f'La casa con {cuartos} cuartos y {distancia} km tiene un valor de ${output}K')

if __name__ == "_main_":
    app.run()
