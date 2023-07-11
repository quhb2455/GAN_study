import numpy as np

# python 내장 함수로 단위 테스트를 할 때 사용함
import unittest
# 약한참조를 만들어서 메모리를 효율적으로 관리할 수 있게 해줌
import weakref
# with 모드를 사용할 수 있도록 해줌
import contextlib

class Variable:
    # name 변수를 추가해서 각 변수를 관리해줌
    def __init__(self, data, name=None) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray) :
                raise TypeError(f'{type(data)}는 취급하지 않음ㅋ')
            
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    # __len__이라는 특수 메소드를 통해서 Variable 인스턴스에 len()을 적용할 수 있게됨
    def __len__(self):
        return len(self.data)

    # __repr__이라는 특수 메소드를 통해서 Variable 인스턴스를 print()로 출력할 수 있게됨
    def __repr__(self):
        if self.data is None :
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n'+' ' * 9)
        return 'variable('+ p +')'

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
        
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    # property를 통해서 함수를 변수처럼 사용할 수 있게됨.
    @property
    def shape(self):
        return self.data.shape           
    @property
    def ndim(self):
        return self.data.ndim
    @property
    def size(self):
        return self.data.size
    @property
    def dtype(self):
        return self.data.dtype


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop :
            self.generation  = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self) 
            
            self.inputs = inputs
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

class Mul(Function) :
    def forward(self, x0, x1):
        y = x0 * x1 
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

# 역전파 활성&비활성 모드를 조절하기 위해 생성
# class 를 인스턴스화 하지 않고 클래스로써만 사용, config를 관리하는것이기 때문에 헷갈리면안됌.
class Config:
    enable_backprop = True

# with구문을 사용하기 위해 선언하는 함수
@contextlib.contextmanager
def using_config(name, value):
    # with 구문에 들어가기 전 사용되는 전처리를 부분
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try :
        yield
    finally :
        # with 구문에서 빠져나갈 때 사용되는 후처리 부분
        setattr(Config, name, old_value)

# unsing_config(~~) 라고 적기 너무 길기고 직관성이 없기 때문에 no_grad라는 함수 생성
def no_grad():
    return using_config('enable_backprop', False)

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

def mul(x0, x1):
    return Mul()(x0, x1)
    

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
    
    a = Variable(np.array(3.0))
    b = Variable(np.array(2.0))
    c = Variable(np.array(1.0))
    
    y = add(mul(a, b), c)
    y.backward()
    
    print(y)
    print(a.grad)
    print(b.grad)
  