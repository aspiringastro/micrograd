# micrograd
Learning Backpropagation one step at a time (with inspiration from @karpathy)


```
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813715870195432, label='b')
    x1w1 = x1*w1
    x2w2 = x2*w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b; n._label = 'n'
    o = n.tanh() 
    o._label = 'o'
    o.backward()
    draw_dot(o)
```

![single pass backward propagation ](https://github.com/aspiringastro/micrograd/blob/main/weight-fwd-prop.png)
