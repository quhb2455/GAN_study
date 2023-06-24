import numpy as np

class Variable:
    def __init__(self, data) -> None:
        # 입력되는 값이 항상 넘파이 값으로 고정
        if data is not None:
            if not isinstance(data, np.ndarray) :
                raise TypeError(f'{type(data)}는 취급하지 않음ㅋ')
            
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func) :
        self.creator = func

    def backward(self):
        # backward를 시작하기 전에 y.grad = np.array(1)을 주던것을 생략할 수 있음.
        if self.grad is None : 
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs :
            f = funcs.pop() 
            x, y = f.input, f.output 
            x.grad = f.backward(y.grad)
            
            if x.creator is not None :
                funcs.append(x.creator)
                
        
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        
        # y값이 항상 ndim array가 되도록 함
        output = Variable(as_array(y))
        output.set_creator(self) 
        
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy) :
        x = self.input.data
        gx = 2 * x * gy
        return gx
    
class Exp(Function) :
    def forward(self, x) :
        return np.exp(x)   
    
    def backward(self, gy) :
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def square(x) :
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

# 입력되는 값이 scalar라면 ndim array로 바꿔주는 함수
def as_array(x):
    if np.isscalar(x) :
        return np.array(x)
    return x

if __name__ == "__main__" :
    
    x = Variable(np.array(1.0))
    y = square(x)
    y.backward()
    print(x.grad, type(x.grad))
    
    z = np.array(1.0)
    print(type(z))