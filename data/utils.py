from numpy import zeros, arange, ndarray
from numpy.random import randint
from torch import Tensor, randn


def create_random_classes(batch_size: int, num_classes: int):
    #TODO адекватные сообщения об ошибке
    assert isinstance(batch_size, int), TypeError("INT")
    assert batch_size > 0, ValueError("It can't be negative")
    
    assert isinstance(num_classes, int), TypeError("INT")
    assert num_classes > 0, ValueError("It can't be negative")
    
    res_np = zeros((batch_size, num_classes))
    res_np[arange(batch_size), randint(0, num_classes, batch_size)] = 1.
    
    return Tensor(res_np)

def add_noise(expr: ndarray, beta: float = 1.0):
    #TODO doc-string
    res = expr + randn(expr.shape).cuda() * beta
    
    return res
