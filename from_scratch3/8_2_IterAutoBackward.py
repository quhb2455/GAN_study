import numpy as np

class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None
        
    def set_creator(self, func) :
        self.creator = func

    def backward(self):
        funcs = [self.creator] # list에 creator 넣기
        while funcs :
            f = funcs.pop() # list에 있느 마지막 creator pop
            x, y = f.input, f.output # creator의 입력과 출력값 호출
            x.grad = f.backward(y.grad) # creator의 입, 출력 값으로 backward 연산
            
            if x.creator is not None :
                funcs.append(x.creator) # 다음 단계의 creator를 list에 넣어줌
                
        
class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        
        output = Variable(y)
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
    
    # 반복문을 이용한 backward는 재귀 방식보다 확장성이 뛰어나고 메모리 효율이 좋음
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)