import numpy as np
import pickle
from dense import Dense
from activations import Tanh, Sigmoid, Softmax
from losses import mse, mse_prime
from keras.utils import to_categorical

class NeuralNetwork:
    def __init__(self):
        self.network = []
        self.epochs = 100
        self.learning_rate = 0.1
        self.scene = None
        self.animate_weights = False
        self.epoch_anim_interval = 10

    def add(self, layer):
        self.network.append(layer)

    def add_scene(self, scene, animate_weights=False, epoch_anim_interval=10 ):
        self.scene = scene
        self.animate_weights = animate_weights and self.scene is not None
        self.epoch_anim_interval = epoch_anim_interval

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            
            # separate lists of dense only and activation only layers
            layers_dense_only = [l for l in self.network if isinstance(l, Dense)]
            layers_activation_only = [l for l in self.network if not isinstance(l, Dense)]

            # output the state of the network
            # self.state_out()
            
            error = 0

            # length of the x_train, y_train set
            train_data_count = len(x_train)

            for train_data_counter, (x, y) in enumerate(zip(x_train, y_train)):

                # # display the current training data in grid
                # if self.scene is not None:
                #     self.scene.go_animate_grid(x)

                # forward
                output = x
                for layer in self.network:
                    output = layer.forward(output)

                    # animate the neurons of the activation layers
                    if (
                        self.scene is not None and
                        not isinstance(layer, Dense) and
                        # train_data_counter == train_data_count - 1 and
                        train_data_counter % 10000 == 0 and
                        (
                         epoch % self.epoch_anim_interval == 0 or
                         epoch == self.epochs - 1
                        )
                    ):
                        layer_index = layers_activation_only.index(layer)
                        if layer_index == 0:
                            self.scene.go_animate_grid(x)
                        self.scene.go_animate_neurons(layer_index, output)
                        print(f"Animated fwd - epoch:{epoch+1}/{epochs}, train_data_counter:{train_data_counter}/{train_data_count}, layer_index:{layer_index}")

                # error
                error += mse(y, output)

                # backward
                grad = mse_prime(y, output)

                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)
                    
                    # animate the weights of the dense layers
                    if (
                        self.animate_weights and
                        isinstance(layer, Dense) and
                        train_data_counter == train_data_count - 1 and
                        (
                         epoch % self.epoch_anim_interval == 0 or
                         epoch == self.epochs - 1
                        )
                    ):
                        layer_index = layers_dense_only.index(layer)
                        self.scene.go_animate_weights(layer_index, layer.weights)
                        print(f"Animated bck - epoch:{epoch+1}/{epochs}, train_data_counter:{train_data_counter}/{train_data_count}, layer_index:{layer_index}")

            error /= len(x_train)
            print(f'{epoch + 1}/{epochs}, error={error}')

    def state_out(self):

        # for count, layer in enumerate(self.network):
        #     if isinstance(layer, Dense):
        #         print(f"Layer: {count} :: {layer.weights}")

        state = []
        for count, layer in enumerate(self.network):
            if isinstance(layer, Dense):
                layer_state = {
                    "weights": layer.weights.tolist(),
                    "biases": layer.bias.tolist()
                }
                print(f"Layer: {count} :: {layer.weights}")
                state.append(layer_state)
        return state

    def predict(self, input_data):

        # separate lists of dense only and activation only layers
        layers_dense_only = [l for l in self.network if isinstance(l, Dense)]
        layers_activation_only = [l for l in self.network if not isinstance(l, Dense)]

        output = input_data
        for layer in self.network:
            output = layer.forward(output)

            # animate the neurons of the activation layers
            if (
                self.scene is not None and
                not isinstance(layer, Dense)
            ):
                layer_index = layers_activation_only.index(layer)
                if layer_index == 0:
                    self.scene.go_animate_grid(input_data)
                self.scene.go_animate_neurons(layer_index, output)

        return output

    def save(self, filename):
        with open(filename, 'wb') as file:

            pickle.dump(self.network, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.network = pickle.load(file)

    def preprocess_data(self, x, y, limit):
        # reshape and normalize input data
        x = x.reshape(x.shape[0], 28 * 28, 1)
        x = x.astype("float32") / 255
        # encode output which is a number in range [0,9] into a vector of size 10
        # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y = to_categorical(y)
        y = y.reshape(y.shape[0], 10, 1)
        return x[:limit], y[:limit]

# Usage example:
if __name__ == "__main__":
    from keras.datasets import mnist
    from keras.utils import to_categorical

    nn = NeuralNetwork()
    nn.add(Dense(28 * 28, 40))
    nn.add(Sigmoid())
    nn.add(Dense(40, 30))
    nn.add(Sigmoid())
    nn.add(Dense(30, 20))
    nn.add(Sigmoid())
    nn.add(Dense(20, 10))
    nn.add(Sigmoid())

    # load MNIST from server
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 3000)
    x_test, y_test = preprocess_data(x_test, y_test, 100)

    # train
    nn.train(x_train, y_train, epochs=300, learning_rate=0.3)

    # test
    for x, y in zip(x_test, y_test):
        output = nn.predict(x)
        print('pred:', np.argmax(output), '\ttrue:', np.argmax(y), '\tcorrect:', np.argmax(output) == np.argmax(y))

    # nn.save('example_trained_model.pkl')
    
    # To load the model
    # nn.load('example_trained_model.pkl')