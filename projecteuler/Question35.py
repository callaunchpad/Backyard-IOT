from math import sqrt, ceil
from datetime import datetime

z = datetime.now()

def is_prime(n):
    if n%2 == 0:
        return False
    else:
        for i in range(3,ceil(sqrt(n))):
            if n%i == 0:
                return False
    return True

def convert(list):
    number = sum(d * 10**i for i, d in enumerate(list[::-1]))
    return number

def circularprime(n):
    ticker = 0
    combolist = [int(i) for i in str(n)]
    while ticker < len(str(n))+1:
        allbutone = combolist[1:]
        one = combolist[0:1]
        combolist = allbutone + one
        newnum = convert(combolist)
        print(newnum)
        print(is_prime(newnum))
        if not is_prime(newnum):
            return False
        ticker += 1
    return True

many = 4

for i in range(11,1000000,2):
    if is_prime(i):
        if circularprime(i):
            many += 1



print(many)

print(datetime.now() - z)
