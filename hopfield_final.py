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
        state=corrupted_pattern.copy() 
        for _ in range(max_steps):
            new_state = self.update(state)
            if np.array_equal(new_state, state): ##(for the dimension array..)
                break
            state = new_state
        return state 
    


##for hopfield with parameter testing
class PerturbedHopfield(HopfieldNetwork):
    """
    Hopfield Network with controlled perturbations for degradation experiments.
    Neutral terminology: weight_decay, retrieval_noise, pattern_overload
    """
    
    def __init__(self, num_neurons):
        super().__init__(num_neurons)
        self.weight_decay_factor = 1.0 ##(since decay degrades)
    
    def apply_weight_decay(self, decay_factor):
        """Reduce synaptic strength: W -> W * decay_factor"""
        if self.original_weights is None:
            raise RuntimeError("Train network first")
        self.weight_decay_factor = decay_factor
        self.weights = self.original_weights * decay_factor
        np.fill_diagonal(self.weights, 0)
    
    def add_retrieval_noise(self, state, flip_probability):
        """Flip bits stochastically during retrieval."""
        if flip_probability <= 0:
            return state.copy()
        noisy = state.copy()
        n_flips = int(len(state) * flip_probability)
        if n_flips > 0:
            idx = np.random.choice(len(state), n_flips, replace=False)
            noisy[idx] *= -1
        return noisy
    
    def retrieve_with_perturbations(self, corrupted_pattern, retrieval_noise=0.0):
        """Retrieve pattern with optional noise injection during recall."""
        state = corrupted_pattern.copy()
        for _ in range(100):
            if retrieval_noise > 0:
                state = self.add_retrieval_noise(state, retrieval_noise)
            new_state = self.update(state)
            if np.array_equal(new_state, state):
                break
            state = new_state
        return new_state