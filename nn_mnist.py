from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from network import NeuralNetwork
from dense import Dense
from activations import Sigmoid

# create neural network
nn = NeuralNetwork()
nn.add(Dense(28 * 28, 36))
nn.add(Sigmoid())
nn.add(Dense(36, 25))
nn.add(Sigmoid())
nn.add(Dense(25, 16))
nn.add(Sigmoid())
nn.add(Dense(16, 10))
nn.add(Sigmoid())

# load MNIST data from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = nn.preprocess_data(x_train, y_train, 60000)
x_test, y_test = nn.preprocess_data(x_test, y_test, 10000)

# train
nn.train(x_train, y_train, epochs=6, learning_rate=0.1)

# To load the model, you can use:
# nn.load('trained_model.pkl')

# test
for x, y in zip(x_test, y_test):
    output = nn.predict(x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y), '\tcorrect:', np.argmax(output) == np.argmax(y))

# print test outcome
accuracy = 0
for x, y in zip(x_test, y_test):
    output = nn.predict(x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y), '\tcorrect:', np.argmax(output) == np.argmax(y))
    accuracy += np.argmax(output) == np.argmax(y)
# print test accuracy
print('accuracy:', accuracy / len(x_test))


# save the trained model
# nn.save('trained_model.pkl')
