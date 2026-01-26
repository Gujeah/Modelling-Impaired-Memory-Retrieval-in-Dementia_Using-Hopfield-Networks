import numpy as np 
import matplotlib.pyplot as plt 


class HopfieldNetwork:

    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))
        self.original_weights = None
##### training our network using hebbian learning 
    def train (self, patterns):
        n_patterns, n_neurons=patterns.shape
        if n_neurons!=self.num_neurons:
            raise ValueError("patterns mismatch") 
        
        print(f"Storing {n_patterns} patterns...")

        W = patterns.T @ patterns
        ##### for recurrent neuronetowrk it cant connect to itsself so the diagonral should be canceeled 
        np.fill_diagonal(W,0)

        self.weights= W/self.num_neurons
        self.original_weights=self.weights.copy()


    def update(self, state, synchronous=False):
         """
        Asynchronous update (default): Neurons update one-by-one in random order.
        Why asynchronous? Guarantees convergence to stable state (energy strictly decreases).
        Synchronous updates can oscillate and fail to converge. 
        calculating the neurons new state based on the same old state, then flip together
        """
         
         if synchronous:
            activations = self.weights @ state
            new_state = np.sign(activations)
            new_state[new_state == 0] = state[new_state == 0]
            return new_state
         else:
             new_state = state.copy()
             for i in np.random.permutation(self.num_neurons):
                ######## since in the human brain neurons fire randomly not sequentially so permutation helps as simulation
                activation = np.dot(self.weights[i], new_state)
                if activation > 0:
                    new_state[i] = 1
                elif activation < 0:
                    new_state[i] = -1

             return new_state

    def predict(self, corrupted_pattern, max_steps=100, synchronous=False):
        pass