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
    # input에 *를 붙여서 리스트로 받음
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        # forward input에도 *를 붙여서 간소화
        ys = self.forward(*xs)
        
        # 결과값이 튜플이 아니라면 튜플로 만들어줘서 다음 연산이 진행될 수 있또록함
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self) 
        
        self.input = input
        self.output = outputs
        # outputs가 리스트 이면 리스트로, 아니면 그냥 스칼라로 
        return outputs if len(outputs) > 1 else outputs[0]
    
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

class Add(Function) :
    # 인수를 각각 하나씩 받음
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)
    

    
def as_array(x):
    if np.isscalar(x) :
        return np.array(x)
    return x

def square(x) :
    f = Square()
    return f(x)

def add(x0, x1):
    return Add()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
    

# 단위 테스트 용 클래스
class SquareTest(unittest.TestCase) :
    def test_forward(self) :
        x = Variable(np.array(2.0))
        y = square(x)
        # 예상값을 하드코딩하여 넣어주고 연산과 같은지 확인함
        excepted = np.array(4.0)
        self.assertEqual(y.data, excepted)
    
    def test_backward(self) :
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        # 예상값 하드코딩
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
    
    def test_gradient_check(self) :
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)
        
if __name__ == "__main__" :
    inputs1 = Variable(np.array(2))
    inputs2 = Variable(np.array(3))
    f = Add()
    ys = f(inputs1, inputs2)
    print(ys.data)

    # 함수로 호출
    y = add(inputs1, inputs2)
    print(y.data)
    
    