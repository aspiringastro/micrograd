import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        # self.data = data.data if isinstance(data, Value) else data
        assert isinstance(data, Value) == False, "attempt to initialize Value with its own type " + str(data) 
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self._label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ops_l = '+'
        out_label = f'{self._label}{ops_l}{other._label}' if (self._label != '' and other._label != '') else '' 
        out = Value(self.data + other.data, (self,other), ops_l,out_label)
        def _backward():
            # Gradient with addition operations is just propagation of the out gradient 
            # to self and other
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ops_l = '*'
        out_label = f'{self._label}{ops_l}{other._label}' if (self._label != '' and other._label != '') else '' 
        out = Value(self.data * other.data, (self,other), ops_l, out_label)
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "pow ops supported only on int or float scalar values"
        ops_l = '**'
        out =  Value(self.data**other, (self,), ops_l, f'{ops_l}{other}')
        def _backward():
            # Gradient on a^n is n * a ^ (n-1) and chain rule
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        assert isinstance(x, Value) == False, "Value object stored as scalar data" + str(self) + str(x)
        t = ((math.exp(2*x)-1)/(math.exp(2*x)+1))
        ops_l = 'tanh'
        out = Value(t, (self,), ops_l, f'{ops_l} {self._label}' )
        def _backward():
            # Gradient on tanh is 1 - tanh^2(x)
            self.grad += (1.0 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        ops_l = 'exp'
        out = Value(math.exp(x), (self,), ops_l, f'{ops_l} {self._label}')
        def _backward():
            # Gradient on exp is exp itself.
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        ops_l = 'ReLU'
        out = Value(0 if self.data < 0 else self.data, (self,), ops_l, f'{ops_l} {self._label}')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

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

        # go one variable at a time and apply chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
        