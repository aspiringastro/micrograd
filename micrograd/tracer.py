
from graphviz import Digraph


def draw_dot(root,_format='svg',_rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert _format in [ 'svg', 'png']
    assert _rankdir in ['LR', 'RL']

    def trace(root):
        # builds a set of all nodes and edges in a graph
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(root)
        return nodes, edges

    nodes, edges = trace(root)
    dot = Digraph(format=_format, graph_attr={'rankdir': _rankdir})
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label = "{ %s| data %.4f | grad %.4f }" % (n._label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid +n._op, uid)
    for e1, e2 in edges:
        dot.edge(str(id(e1)), str(id(e2)) + e2._op)
    return dot