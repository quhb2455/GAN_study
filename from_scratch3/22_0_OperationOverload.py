import numpy as np

# python 내장 함수로 단위 테스트를 할 때 사용함
import unittest
# 약한참조를 만들어서 메모리를 효율적으로 관리할 수 있게 해줌
import weakref
# with 모드를 사용할 수 있도록 해줌
import contextlib

class Variable:
    # 연산자 우선 순위를 높여주는 것. 이것을 통해서 좌, 우항에 상관없이 Variable 함수의 연산자가 호출됨
    __array_priority__ = 200
    
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

    # __mul__ 이라는 특수 메소드를 통해서 * 기호로 연산할 수 있게 해줌
    def __mul__(self, other):
        # y = a * b 라고하면 * 연산자 왼쪽에 있는 a가 self 에 전달되고 오른쪽에 있는 b가 other에 전달됨
        return mul(self, other)
        
    
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
        # 입력되는 변수가 Variable이라면 그대로 진행하고 아니라면 Variable 형태로 변경시켜줌
        inputs = [as_variable(x) for x in inputs]
        
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

# 입력되는 값의 부호를 변경해주는 클래스
class Neg(Function) :
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

# 뺄셈 추가
class Sub(Function) :
    def forward(self, x0, x1):
        return x0 - x1
    
    # 뺄셈의 미분값은 x0은 1 이고 x1 은 -1임
    def backward(self, gy):
        return gy, -gy

# 나눗셈 추가
class Div(Function) :
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy *(-x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function) :
    def __init__(self, c) -> None:
        self.c = c
        
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

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

# 함수에 입력되는 값을 Variable로 변환하기 위함
def as_variable(obj):
    if isinstance(obj, Variable) :
        return obj
    return Variable(obj)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

def square(x) :
    f = Square()
    return f(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

# 부호 변경 함수 추가
def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

# 앞뒤가 변경됐을 때 연산하는 거 설정
def rsub(x0, x1) :
    x1 = as_array(x1)
    return Sub()(x1, x0)#입력의 반대로

def div(x0, x1) :
    x1 = as_array(x1)
    return Div()(x0, x1)

# 앞 뒤가 변경됐을 떄 연산하는 거 설정
def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) #입력의 반대로

# 거듭제곱 연상 추가
# 역방향 거듭제곱은 구현안함
def pow(x, c):
    return Pow(c)(x)

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
    Variable.__mul__ = mul
    Variable.__add__ = add
    # 3.0 * Variable(np.array(1.0))과 같이 반대 상황을 위해 rmul과 radd를 선언
    Variable.__rmul__ = mul
    Variable.__radd__ = add
    # 부호를 변경해주는 클래스 설정
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    # 앞뒤가 바뀌었을 때 연산하는 것을 설정
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    
    # as_variable을 통해서 연산가능
    a = Variable(np.array(3.0))
    y = a + np.array(3.0)    
    print(y)
    
    # as_array를 통해서 연산 가능
    a = Variable(np.array(3.0))
    y = a + 4.0
    print(y)
    
    # rmul과 radd 때문에 Variable의 위치에 관계없이 연산가능
    a = Variable(np.array(2.0))
    y = 3.0 * a + 4.0
    print(y)
    
    # __array_priority__ 덕분에 연산자 우선 순위가 높여졌고 따라서 np.array의 연산자보다 Variable의 연산자가 먼저 호출됨
    a = Variable(np.array(2.0))
    y = np.array(2.0) * a
    print(y)
    
    # 음수로 변경할 수 있음
    a = Variable(np.array(-20))
    y = -a
    print(y)
    
    # 뺄셈과 역방향 뺄셈 가능
    a = Variable(np.array(20))
    y1 = 29 - a
    y2 = a - 29
    print(y1, y2)
    
    # 역방향 거듭제곱은 구현안함
        # ex) 2 ** a 와 같은 구조
    a = Variable(np.array(30))
    y = a ** 2
    print(y)
    
