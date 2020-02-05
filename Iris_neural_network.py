
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense
import tensorflow as tf


def preprocess_data():
    data_name = 'Data/iris.data'
    data = pd.read_csv(data_name, sep=',', header=None)
    data.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class']

    # mix the observations
    data = shuffle(data)
    data.reset_index(inplace=True, drop=True)

    # train one-hot encoding
    values = np.array(data['Class'])
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = values.reshape(len(values), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y_train = onehot_encoded[:110, :]
    y_val = onehot_encoded[110:130, :]
    y_test = onehot_encoded[130:, :]

    x_values = data.values[:, :4]
    x_train = x_values[:110, :]
    x_val = x_values[110:130, :]
    x_test = x_values[130:, :]

    print(onehot_encoder.categories_)

    return (x_train, x_val, x_test), (y_train, y_val, y_test)


def train_network(x_data, y_data):
    model = Sequential()
    model.add(Dense(6, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    model.fit(x_data[0], y_data[0],
              epochs=200,
              validation_data=(x_data[1], y_data[1]))

    # evaluate the keras model
    _, accuracy = model.evaluate(x_data[2], y_data[2])

    print('----------------------------------')
    print('Accuracy: ', accuracy)

    return model


def load_local_model():
    # read the model from the serialized JSON file
    """
    json_file = open('Models/Iris_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into the new model
    loaded_model.load_weights('Models/Iris_model_weights.h5')

    print('Loaded the model from the disk')

    # compile the loaded model
    loaded_model.compile('adam', loss='categorical_crossentropy')
    """
    loaded_model = load_model('Models/Iris_model.h5')
    graph = tf.get_default_graph()

    return loaded_model, graph


# convert a json query??? to a numpy array
def save_model(input_model):
    # to JSON
    model_json = input_model.to_json()
    with open('Models/Iris_model.json', 'w') as json_file:
        json_file.write(model_json)
    input_model.save_weights('Models/Iris_model_weights.h5')


if __name__ == '__main__':
    path = 'Models/Iris_model.h5'

    # x, y = preprocess_data()
    # model = train_network(x, y)
    # model.save('Data/Iris_model.h5')

    model = load_model(path)
    save_model(model)

