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
    
    # 동일한 인스턴스로 다른 연산에 2번 이상 이용하면 grad가 덮어지는 문제를 해결하기 위해 추가
    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None : 
            self.grad = np.ones_like(self.data)
            
        funcs = [self.creator]
        while funcs :
            f = funcs.pop() 

            gys = [output.grad for output in f.output]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            # 원본 코드로 진행하게되면 값을 계속 덮어쓰기 때문에 변경. 
            for x, gx in zip(f.inputs, gxs) :
                # x.grad가 처음 호출되면 항상 None 값을 가지므로 처음 호출 될 때에 gx를 넣음
                if x.grad is None :
                    x.grad = gx
                else :
                    # x.grad에 값이 있다면 동일한 인스턴스이므로 값을 더해줌.
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)
                
        
class Function:
    # input에 *를 붙여서 리스트로 받음
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        
        if not isinstance(ys, tuple) :
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self) 
        
        self.inputs = inputs
        self.output = outputs
        
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, gy):
        raise NotImplementedError
    
class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy) :
        # 입력이 항상 튜플로 오기 때문에 입력이 하나만 필요한 square의 경우 0번째 인덱스만 사용
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
    inputs1 = Variable(np.array(2))
    # inputs2 = Variable(np.array(3))
    
    z = add(square(inputs1), square(inputs1))
    z.backward()

    print(z.data)
    print(inputs1.grad)

    # 위에서 사용한 inputs1을 그대로 다른 연산에 사용
    # grad 결과값이 위에서 한 값에 더해져서 나옴. 그래서 부정확함
    z = add(add(inputs1, inputs1), inputs1)
    z.backward()
    print(z.data)
    print(inputs1.grad)

    # grad 값을 초기화해줌
    inputs1.cleargrad()
    # inputs1 를 다시 다른 연산에 사용
    # grad 값이 제대로 출력됨
    z = add(add(inputs1, inputs1), inputs1)
    z.backward()
    print(z.data)
    print(inputs1.grad)

    # print(inputs2.grad)
    