# Neural Network from Scratch
  
## Cloned from: Neural Network From Scratch - by [Independent Code (Omar Aflak)](github.com/TheIndependentCode)
  
Modified by turning network (`network.py`) into an object
  
## How to build neural network
- a `layer` is defined as a base class with `forward` and `backward` propagation methods
- `activation` and `dense` classes inherit from `layer` class
- a `network` class is instantiated
- a `neural net` is formed by adding layers (i.e., `dense` and `activation` instances) to the instantiated `network` 
- see `mnist.py` for example 
  
   
## Changes from original
- changed mnist.py to work with tensorflow on macos/metal
- some activations converted to lambdas
- converted functional flow into object based model
  

  
# ----- ORIGINAL README FOLLOWS -----
  
  
# Neural Network From Scratch

This code is part of my video series on YouTube: [Neural Network from Scratch | Mathematics & Python Code](https://youtube.com/playlist?list=PLQ4osgQ7WN6PGnvt6tzLAVAEMsL3LBqpm).

# Try it!

```
python3 xor.py
```

# Example

```python
import numpy as np

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime
from network import train

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

train(network, mse, mse_prime, X, Y, epochs=10000, learning_rate=0.1)
```