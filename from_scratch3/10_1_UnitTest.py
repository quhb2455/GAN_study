import numpy as np

# python 내장 함수로 단위 테스트를 할 때 사용함
import unittest

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray) :
                raise TypeError(f'{type(data)}는 취급하지 않음ㅋ')
            
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func) :
        self.creator = func

    def backward(self):
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

def as_array(x):
    if np.isscalar(x) :
        return np.array(x)
    return x

# 단위 테스트 용 클래스
class SquareTest(unittest.TestCase) :
    def test_forward(self) :
        x = Variable(np.array(2.0))
        y = square(x)
        # 예상값을 하드코딩하여 넣어주고 연산과 같은지 확인함
        excepted = np.array(4.0)
        self.assertEqual(y.data, excepted)
        
if __name__ == "__main__" :
    unittest.main()