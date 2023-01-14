import numpy as np

#1 посчитать значений производной функции $\cos(x) + 0.05x^3 + \log_2{x^2}$ в точке $x = 10$. 
#Ответ округлить до 2-го знака. Пожалуйста назовите функцию derivation, функция должна принимать точку, в 
#которой нужно вычислить значение производной, и функцию, производную которой мы хотим вычислить.

dx = 0.00001

#$\cos(x) + 0.05x^3 + \log_2{x^2}$
def func1(x):
    return np.cos(x) + 0.05 * x**3 + np.log2(x**2)

def derivation(x, func):
    return round((float(func(x + dx) - func(x)) / dx), 2)

#x1 = 10
#print(derivation(x1, func1))    

#2 посчитать значение градиента функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$ в точке $(10, 1)$. 
#Пожалуйста назовите функцию gradient, функция должна принимать список с координатами точки, 
#в которой нужно вычислить значение производной, и функцию, производную которой мы хотим вычислить. Ответ округлить до 2-го знака.
def func2(x):
    return x[0]**2 * np.cos(x[1]) + 0.05 * x[1]**3 + 3 * x[0]**3 * np.log2(x[1]**2)   

def gradient(x, func):
    return [round(float((func([x[0] + dx, x[1]]) - func(x))) / dx, 2),
            round(float((func([x[0], x[1] + dx]) - func(x))) / dx, 2)]

#twoX = [10, 1] 
#print(gradient(twoX, func2))

#3 найти точку минимуму для функции $\cos(x) + 0.05x^3 + \log_2{x^2}$.
#Зафиксировать параметр $\epsilon = 0.001$, начальное значение принять равным 10. 
#Выполнить 50 итераций градиентного спуска. Ответ округлить до второго знака; 
#Пожалуйста назовите функцию gradient_optimization_one_dim. Функция должна принимать на вход функцию, которую требуется оптимизировать.
epsilon = 0.001
iterations = 50

def gradient_optimization_one_dim(func):
    x = 10
    for i in range(iterations):
        x = x - epsilon * derivation(x, func)
    return round(x, 2)

#print(gradient_optimization_one_dim(func1))

#4 найти точку минимуму для функции $x_1^2\cos(x_2) + 0.05x_2^3 + 3x_1^3\log_2{x_2^2}$. 
#Зафиксировать параметр $\epsilon = 0.001$, начальные значения весов принять равным [4, 10]. s
#Выполнить 50 итераций градиентного спуска. Ответ округлить до второго знака; Пожалуйста назовите функцию gradient_optimization_multi_dim.
def gradient_optimization_multi_dim(func):
    x1 = 4
    x2 = 10
    for i in range(iterations):
        funcGradient = gradient([x1, x2], func)
        x1 = x1 - round(epsilon * funcGradient[0], 2)
        x2 = x2 - round(epsilon * funcGradient[1], 2)
    return [round(x1, 2), round(x2, 2)]