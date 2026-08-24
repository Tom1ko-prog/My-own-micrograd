import math


class Value:
    def __init__(
        self, data, _children: tuple = (), _op: str = "", label: str = ""
    ) -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"**{other}")

        def backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = backward
        return out

    def sqrt(self):
        out = Value(math.sqrt(self.data), (self,), "sqrt")

        def backward():
            self.grad += (0.5 / (out.data + 1e-7)) * out.grad

        out._backward = backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def backward():
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def log(self):
        x_clamped = max(min(self.data, 1 - 1e-12), 1e-12)
        out = Value(math.log(x_clamped + 1e-7), (self,), "log")

        def backward():
            self.grad += (1 / (x_clamped + 1e-7)) * out.grad

        out._backward = backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = backward
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = backward
        return out

    def ReLU(self):
        out = Value(max(self.data, 0), (self,), "ReLU")

        def backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = backward
        return out

    def LeakyReLU(self, alpha: float = 0.01):
        x = self.data
        out = Value(max(alpha * x, x), (self,), "LeakyReLU")

        def backward():
            self.grad += (alpha if x < 0 else 1) * out.grad

        out._backward = backward
        return out

    def ELU(self, alpha: float = 0.01):
        x = self.data
        e = math.exp(x)
        out = Value(alpha * (e - 1) if x <= 0 else x, (self,), "ELU")

        def backward():
            self.grad += (alpha * e if x < 0 else 1) * out.grad

        out._backward = backward
        return out

    def GELU(self):
        inner = self + 0.044715 * self**3

        tanh_out = (inner * math.sqrt(2 / math.pi)).tanh()
        cdf = (tanh_out + 1) * 0.5

        return self * cdf

    def SiLU(self):
        x = self.data
        s = 1 / (1 + math.exp(-x))
        out = Value(x * s, (self,), "SiLU")

        def backward():
            self.grad += (s + x * s * (1 - s)) * out.grad

        out._backward = backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)
