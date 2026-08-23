import random

from graph import *
from value import *


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin: int) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x) -> Value:
        out = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, act=None) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.act = act

    def __call__(self, x):
        if self.act:
            outs = [self.act(n(x)) for n in self.neurons]
        else:
            outs = [n(x) for n in self.neurons]

        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, acts=None) -> None:
        sz = [nin] + nouts

        if acts is None:
            acts = [None] * len(nouts)

        self.layers = [Layer(sz[i], sz[i + 1], act=acts[i]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
