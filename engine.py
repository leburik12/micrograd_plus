
class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self) -> 'Value':
        """Sigmoid activation function
        
        Mathematical definition
            σ(x) = 1 / (1 + e^-x)
            
        Derivative
            dσ/dx = σ(x)*(1 - σ(x))
        """
        x = self.data
        if x < -20: # Avoid underflow, sigmoid = 0
            out = Value(0.0, (self,), 'sigmoid')   
            def _backward():
                self.grad += 0.0 * out.grad
        elif x > 20: # sigmoid = 1
            out = Value(1.0, (self,), 'sigmoid')
            def _backward():
                self.grad += 0.0 * out.grad
        else:  
            # For x∈[−20,20], sigmoid is numerically stable.
            ez = math.exp(-x)
            s = 1 / (1 + ez)
            
            out = Value(s, (self,), 'sigmoid')
            def _backward():
                self.grad += (s * (1 - s)) * out.grad
                
            out._backward = _backward
            return out
        
    def leaky_relu(self, alpha: float = 0.01) -> 'Value':
        """Leaky ReLU activation
        
        Mathematical definition
            LReLU(x) = x if x > 0 else αx
            
        Derivative: 
            dLReLU/dx = 1 if x > 0 else α
        """
        out = Value(self.data if self.data > 0 else alpha * self.data,
                    (self,), f'LeakyReLU({alpha})')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else alpha) * out.grad
        out._backward = _backward
        
        return out
    
    def elu(self, alpha: float = 1.0) -> 'Value':
        """Exponential Linear Unit activation
           
        Mathematical definition:
            ELU(x) = x if x > 0 else α(e^x - 1)   
            
        Derivative:
            dELU/dx = 1 if x > 0 else ELU(x) + α
        """
        x = self.data
        if x > 0:
            out = Value(x, (self,), f'ELU({alpha})')
        else:
            expx = math.exp(x)
            out = Value(alpha * (expx - 1), (self,), f'ELU({alpha})')
            def _backward():
                self.grad += (out.data + alpha) * out.grad
        out._backward = _backward
        return out
    
    def swish(self, beta: float = 1.0) -> 'Value':
        """Swish activation function (self-gated)
           
        Mathematical definition: 
            Swish(x) = x * σ(βx)
            where σ is the sigmoid function
            
        Derivative:
            dSwish/dx = Swish(x) + σ(βx)(1 - Swish(x))
        """
        x = self.data
        b = beta 
        
        # Compute sigmoid(bx) using stable implementation
        if b*x < -20:
            sig = 0.0
        elif b*x > 20:
            sig = 1.0
        else:
            if b*x > 0:
                ez = math.exp(-b*x)
                sig = 1 / (1 + ez)
            else:
                ez = math.exp(b*x)
                sig = ez / (1 + ez)
                
        out = Value(x * sig, (self,), f'Swish({beta})')
        
        def _backward():
            self.grad += (out.data + sig * (1 - out.data)) * out.grad
        out._backward = _backward
        
        return out
    
    def gelu(self) -> 'Value':
        """ Gaussian Error Linear Unit activation.
        
        Mathematical appromixation
            GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            
        Derivatieve func:
        """
        
        x = self.data
        x = self.data
        
        sqrt_2_over_pi = math.sqrt(x / math.pi)
        cdf = 0.5 * (1 + maht.tanh(sqrt _2_over_pi, (x + 0.044715 * x**3)))
        out = Value(x * cdf, (self,), 'GELU')
        
        def _backward():
            #  Approximate gradient of GELU
            gelu_grad = cdf + x * (1 - cdf**2) * sqrt_2_over_pi * (1 + 0.134145 * x**2)
            self.grad += gelu_grad * out.grad
        out._backward = _backward
        
        return out
    
    def softplus(self, beta: float = 1.0) -> 'Value':
        """Softplus activiation function
        
        Mathematical definition:
           Softplus(x) = 1/β * log(1 + e^(βx))
        
        Derivative:
           dSoftplus/dx = σ(βx)
             where σ is sigmoid function
        """
        
        x = self.data
        b = beta 
        
        if b*x > 20:
            out = Value(x, (self,), f'Softplus({beta})')
            
            def _backward():
                self.grad += 1.0 * out.grad
        else:
            out = Value(math.log(1 + math.exp(b*x))/b, (self,), f'Softplus({beta})')
            def _backward():
                self.grad += (1 / (1 + math.exp(-b*x))) * out.grad
        out._backward = _backward
        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"