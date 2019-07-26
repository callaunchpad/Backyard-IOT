def c():
    return True or False


def if_function(condition, true_result, false_result):
    if condition:
        return true_result
    else:
        return false_result

def f():
    if c() is True:
        return 1
    if c() is False:
        return 2

def with_if_function():
    return if_function(c(), t(), f())

def t():
    if c() is True:
        return 6
    if c() is False:
        return 4

print(with_if_function())
print(c())