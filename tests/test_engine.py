import unittest

from micrograd.engine import Value

class TestMicrogradEngine(unittest.TestCase):
    def test_basic_rstr(self):
        x = Value(2.0)
        print(f"x={x}")

    def test_value_add(self):
        x = Value(2.0)
        y = Value(3.0)
        z =x + 1.0 + y
        print(f"z={z}")

    def test_value_sub(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x - 1.0 - y
        print(f"z={z}")

    def test_value_mul(self):
        x = Value(2.0)
        y = Value(3.0)
        z =x * 1.0 * y
        print(f"z={z}")

    def test_value_pow(self):
        x = Value(3.0)
        y = x**2
        print(f"x = {x}, y = x**2 = {y}")

    def test_value_div(self):
        x = Value(2.0)
        y = Value(3.0)
        z = x / y
        z_r = (2.0 / y)
        print(f"z= {x}/{y} = {z}, z_r={z_r}")
        
    def test_value_tanh(self):
        #inputs x1, x2
        x1 = Value(2.0, label='x1')
        x2 = Value(0.0, label='x2')

        # weights w1, w2
        w1 = Value(-3.0, label='w1')
        w2 = Value(1.0, label='w2')

        # x1w1 + x2w2
        x1w1 = x1*w1 + x2*w2
        L = x1w1.tanh()

    def test_value_backward(self):
        a = Value(-2.0, label='a')
        b = Value(3.0, label='b')
        d = a * b
        e = a + b
        f = d * e
        f.backward()

    def test_value_exp(self):
        x1 = Value(2.0, label='x1')
        x2 = Value(0.0, label='x2')
        w1 = Value(-3.0, label='w1')
        w2 = Value(1.0, label='w2')
        b = Value(6.8813715870195432, label='b')
        x1w1 = x1*w1
        x2w2 = x2*w2
        x1w1x2w2 = x1w1 + x2w2
        n = x1w1x2w2 + b; n._label = 'n'
        e = (2*n).exp()
        o = (e - 1)/(e + 1)
        o._label = 'o'
        o.backward()
        


    