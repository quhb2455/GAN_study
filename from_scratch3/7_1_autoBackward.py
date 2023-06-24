from typing import Any
import numpy as np

class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None
        
    # 결과값 계산된 위치를 저장하기 위한 것.
    def set_creator(self, func) :
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        
        output = Variable(y)
        # 결과값 계산된 위치를 저장. 
        # output이라는 변수는 set_creator에 의헤서 현재 변수가 속한 인스턴스의 위치를 기록하게됨.
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
    
    
if __name__ == "__main__" :
    A = Square()
    B = Exp()
    C = Square()
    
    x = Variable(np.array(0.5))
    
    a = A(x)
    b = B(a)
    y = C(b)
    
    # 변수 y가 생성된 위치
    assert y.creator == C
    # 변수 y가 생성된 위치의 입력
    assert y.creator.input == b
    # 변수 y가 생성된 위치의 입력값이 생성된 위치 ...
    assert y.creator.input.creator == B
    assert y.creator.input.creator.input == a
    assert y.creator.input.creator.input.creator == A
    assert y.creator.input.creator.input.creator.input == x
    
    print(y)
    
    ## 출력 값의 생성자 -> 입력 -> 생성자 -> 입력... 순을 통해서 출력값 하나로 모든 값들을 호출 할 수 있음.
    # 출력값의 윗 단계로 하나씩 거슬러 올라가는 방식
    
    # 1. 역전파 시작
    y.grad = np.array(1.0)
    # 마지막 출력 값의 creator를 호출하여 마지막 연산을 가져올 수 있음.
    C = y.creator
    # 마지막 연산의 입력값을 가져옴
    b = C.input
    # 마지막 연산의 backward함수를 호출하여 마지막 연산의 입력값에 대한 gradient를 구함.
    b.grad = C.backward(y.grad)
    
    # 2. 역전파 시작
    B = b.creator # 함수 가져오기
    a = B.input # 함수의 입력값 가져오기
    a.grad = B.backward(b.grad) # 함수의 backward 호출
    
    # 3. 역전파 시작
    A = a.creator
    x = A.input
    x.grad = A.backward(a.grad)
    
    print(x.grad)
    
    
    