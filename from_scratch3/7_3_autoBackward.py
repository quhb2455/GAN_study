import numpy as np

class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None
        
    # 결과값 계산된 위치를 저장하기 위한 것.
    def set_creator(self, func) :
        self.creator = func

    def backward(self):
        f = self.creator # 현재 변수의 생성자 호출
        if f is not None :
            x = f.input # 생성자의 입력값 호출
            x.grad = f.backward(self.grad) # 현재 변수의 미분값과 현재 함수의 backward를 호출
            x.backward() # 한 단계 앞 변수의 backward 호출(재귀적 호출)
    
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
    
    # 최종 출력값의 backward를 호출함으로써 전체 변수에 대한 미분이 가능    
    # 최종 출력값에 연결된 다른 값들을 재귀적으로 호출하면서 진행됨.
    # self.creator == None인 변수를 만날 때까지 계속 호출됨.
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)