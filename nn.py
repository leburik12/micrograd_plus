import random
from micrograd+.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
        
    def parameters(self):
        return []
    
class Neuron(Module):
    VALID_ACTIVATIONS = {
        'relu': 'ReLU',
        'sigmoid': 'Sigmoid',
        'leaky_relu': 'LeakyReLU',
        'elu': 'ELU',
        'swish': 'Swish',
        'gelu': 'GELU',
        'softplus': 'Softplus',
        'linear': 'Linear'
    }
     
    def __init__(self, nin, activation='relu', **act_kwargs):
        # Initialize weights using kaiming initialization 
        scale = math.sqrt(2.0/nin)   # good for relu
        
        # Initialize weights and bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        
        # validate and store activation 
        self.activation = activation.lower()
        self._validate_activation()
        self.act_kwargs = act_kwargs
        
    def _validate_activation(self):
        if self.activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{self.activation}'. "
                f"Available: {list(self.VALID_ACTIVATIONS.keys())}"
            )
    
    def __call__(self, x):
        """ Forward pass: activation(w.x + b) """
        
        # Compute weighted sum of inputs
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        
        # Apply activation function
        if self.activation == 'linear':
            return act
        
        # Get the activation method and call it
        activation_method = getattr(act, self.activation)
        return activation_method(**self.act_kwargs)
        
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{self.activation.capitalize()} Neuron ({len(self.w)})" 
        
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"   

class MLP(Module):
    
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"