import numpy as np

class Variable :
    def __init__(self, data) -> None:
        self.data = data

class Function:
    def __call__(self, input) -> None:
        x = input.data

        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
class Exp(Function) :
    def forward(self, x):
        return np.exp(x)

class Square(Function) :
    def forward(self, x):
        return x ** 2
    
if __name__ == "__main__" :
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))

    a = A(x)
    b = B(a)
    c = C(b)
    print(c.data)