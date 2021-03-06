import math

def quadratic(a, b, c):
    p = b * b - 4 * a * c
    if p >= 0 and a != 0:  # 一元二次方程有解的条件
        x1 = (-b + math.sqrt(p)) / (2 * a)
        x2 = (-b - math.sqrt(p)) / (2 * a)
        return x1, x2
    elif a == 0:  # a=0的情况下为一元一次方程
        x1 = x2 = -c / b
        return x1
    else:
        return ('方程无解')

a = float(input('Please input a='))
b = float(input('Please input b='))
c = float(input('Please input c='))
print(quadratic(a, b, c))
