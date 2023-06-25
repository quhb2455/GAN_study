import numpy as np

# python 내장 함수로 단위 테스트를 할 때 사용함
import unittest
# 약한참조를 만들어서 메모리를 효율적으로 관리할 수 있게 해줌
import weakref

class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray) :
                raise TypeError(f'{type(data)}는 취급하지 않음ㅋ')
            
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
        
    def set_creator(self, func) :
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None : 
            self.grad = np.ones_like(self.data)
        
        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set :
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x:x.generation)
        add_func(self.creator)

        while funcs :
            f = funcs.pop() 
            
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs) :
                if x.grad is None :
                    x.grad = gx
                else :
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
        
            # 연산에서 생성되는 모든 output의 grad 값을 비워줌으로써 최초 입력 값의 grad 값만 그대로 남아있게됨
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None
                
        
class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation  = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self) 
        
        self.inputs = inputs
        # Variable -> Function -> Variable 순으로 순환참조를 하는 구조였는데 weark.ref를 통해서 약한참조로 변경
        self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy) :
        x = self.inputs[0].data
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
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)
    
    def backward(self, gy):
        return gy, gy        
    
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
    
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    
    t = add(x0, x1)
    y = add(x0, t)
    
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)