import random
from micrograd.engine import Value

class Module:
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1), label=f'w{n}') for n in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, xin):
        # dot product of w and x +b (w * x + b)
        x = [xin[i] if isinstance(xin[i], Value) else Value(xin[i], label=f'x{i}') for i in range(len(xin))]
        act = sum((wi * xi  for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f'N{len(self.w)}'

class Layer(Module):

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"L[{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [ Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP[{', '.join(str(layer) for layer in self.layers)}]"
