
from flask import Flask, render_template, request
import numpy as np
from Iris_neural_network import load_local_model


# init flask
app = Flask(__name__, template_folder='Templates')

global model, graph
model, graph = load_local_model()


# main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # save the 4 user inputs into a np array
        input = np.zeros((1, 4))
        input[0, 0] = float(request.form['sl'])
        input[0, 1] = float(request.form['sw'])
        input[0, 2] = float(request.form['pl'])
        input[0, 3] = float(request.form['pw'])

        # predict the result using your model
        with graph.as_default():
            res = model.predict(input)
            result = res[0, :].argmax()

        # interpret the results
        classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        response = classes[result]
        return render_template("index.html", result=response)
    return render_template("index.html", content='')


# main function of the program
if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    app.run()



