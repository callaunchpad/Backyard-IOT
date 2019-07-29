from datetime import datetime
from math import sqrt, ceil
z = datetime.now()

def is_prime(n):
    if n == 1:
        return False
    elif n ==2:
        return True
    elif n%2 == 0:
        return False
    for i in range(3,ceil(sqrt(n))+1):
        if n%i == 0:
            return False
    return True

def truncatableleft(n):
    index = len(str(n)) - 1
    n = n%(10**index)
    while n >= 1:
        if not is_prime(n):
            return False
        index = len(str(n)) - 1
        n = n%(10**index)
    return True

def truncatableright(n):
    list = [int(i) for i in str(n)]
    del list[-1]
    index = len(list)
    while index > 0:
        number = sum(d * 10 ** i for i, d in enumerate(list[::-1]))
        if not is_prime(number):
            return False
        del list[-1]
        index = len(list)
    return True

sum1 = 0
for i in range(11,1000000,2):
    if is_prime(i) and truncatableleft(i) and truncatableright(i):
        sum1 += i
print(sum1)

print(datetime.now() - z)