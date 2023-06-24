import numpy as np

class Variable :
    def __init__(self, data) -> None:
        self.data = data

class Function:
    def __call__(self, input) -> None:
        x = input.data
        y = x ** 2

        output = Variable(y)
        return output
    
if __name__ == "__main__" :
    x = Variable(np.array(10))
    f = Function()
    y = f(x)

    print(type(y))
    print(y.data)