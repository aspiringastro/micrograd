import unittest

from micrograd.neuron import Neuron, Layer, MLP
class TestMicrogradNeuron(unittest.TestCase):
    def test_2d_neuron(self):
        x = [ 2.0, 3.0, 0.5, 1.0 ]
        n = Layer(4, 9)
        o = n(x)
        print(o)


    def test_mlp_neuron(self):
        x = [2.0, 3.0, -1.0]
        n = MLP(3, [4, 4, 1])
        n(x)
        
