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

class Square(Function) :
    def forward(self, x) :
        return x ** 2

class Exp(Function) :
    def forward(self, x):
        return np.exp(x)

# 수치미분 공식
def numerical_diff(f, x, eps=1e-4) :
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)

    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

# 합성함수
def f(x) :
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))

if __name__ == "__main__" :
    _f = Square()
    x = Variable(np.array(9.0))

    # 일반 함수 미분
    dy = numerical_diff(_f, x)
    print(dy)

    print("=" * 10)

    # 합성 함수 미분
    x = Variable(np.array(0.5))
    dy = numerical_diff(f, x) 
    print(dy)