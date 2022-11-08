import unittest

from micrograd.engine import Value
from micrograd.tracer import *

class TestMicrograd(unittest.TestCase):

    def test_expression(self):
        a = Value(3.0)
        b = Value(4.0)
        c = Value(5.0)
        d = a * b + c
        draw_dot(d)