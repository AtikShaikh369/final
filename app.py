import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, json
import pickle
import json
import matplotlib.pyplot as plt
import mpld3
from model import X, y

app = Flask(__name__)
model_DT = pickle.load(open('model_DT.pkl', 'rb'))
model_KNN = pickle.load(open('model_KNN.pkl', 'rb'))
model_SVM = pickle.load(open('model_SVM.pkl', 'rb'))
model_NB = pickle.load(open('model_NB.pkl', 'rb'))

#y_pred = model_DT.predict([[6.6, 6.7, 34, 129, 967, 181, 70]])
# y_pred


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction_DT = model_DT.predict(final_features)
        prediction_KNN = model_KNN.predict(final_features)
        prediction_SVM = model_SVM.predict(final_features)
        prediction_NB = model_NB.predict(final_features)

        output_DT = round(prediction_DT[0], 2)
        output_KNN = round(prediction_KNN[0], 2)
        output_SVM = round(prediction_SVM[0], 2)
        output_NB = round(prediction_NB[0], 2)
        
        list = [output_DT, output_KNN, output_SVM, output_NB]
        list = [2, 1, 0, 2]
        max = 0
        res = list[0] 
        for i in list: 
            freq = list.count(i) 
            if freq > max: 
                max = freq 
                res = i 
        quality = find_quality(res)
        output = {'Decision Tree': output_DT, 'KNN': output_KNN,
                  'SVM': output_SVM, 'Naive Bayes': output_NB}

        input = request.form
        return render_template('result.html',  prediction_text=output, quality=quality, input = input)
    else:
        return render_template('predict.html')

def find_quality(argument): 
    switcher = { 
        0: "low", 
        1: "medium", 
        2: "high", 
    } 
    return switcher.get(argument, "nothing")

@app.route('/accuracy')
def accuracy():
    from model import accuracy
    return render_template('accuracy.html', accuracy=accuracy)


@app.route('/datavisual')
def visual():
    return render_template('visual.html')


def draw_fig(plot):
    if plot == 'line':
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4, 5, 6]
        y = [2, 4, 8, 16, 32, 64]
        ax.plot(x, y)
        graph = mpld3.fig_to_html(fig)
        plots = {'graph': graph}
        return plots
    elif plot == 'histogram':
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4, 5, 6]
        y = [2, 4, 6, 8, 10, 12]
        ax.hist(x, y)
        graph = mpld3.fig_to_html(fig)
        plots = {'graph': graph}
        return plots
    elif plot == 'scatter':
        fig, ax = plt.subplots()
        from model import dataset
        ax.hist(dataset['pH'])
        ph = mpld3.fig_to_html(fig)

        fig, ax = plt.subplots()
        ax.hist(dataset['CEC'])
        cec = mpld3.fig_to_html(fig)

        fig, ax = plt.subplots()
        ax.hist(dataset['calcium'])
        calcium = mpld3.fig_to_html(fig)
        plots = {'ph': ph, 'cec': cec, 'calcium': calcium}
        return plots


@app.route('/plot', methods=['POST', 'GET'])
def plot():
    if request.method == 'POST':
        plot = request.form.get('plot')
        plots = draw_fig(plot)
        return render_template('visual.html', plots=plots)
    else:
        return render_template('visual.html')


if __name__ == "__main__":
    app.run(debug=True)
