from datetime import datetime
from math import sqrt,ceil
z = datetime.now()

#Detecting if a number is prime or not
def is_prime(n):
    if n == 2:
        return True
    elif n == 1 or n%2 == 0:
        return False
    else:
        for i in range(3,ceil(sqrt(n))+1,2):
            if n%i == 0:
                return False
    return True

#Iteration through all odd numbers, filtering out those with primes
for i in range(23,30000,2):
    if is_prime(i):
        continue
    else:
        ticker = 0   # If ticker > 0, I would know that there is possible solution of prime and square
        index = 1
        num = i - 2 * (index ** 2)
        #Iteration by subtracting double the square and seeing if remainder is prime
        while num > 0:
            if is_prime(num):
                ticker += 1
                break
            else:
                index += 1
                num = i - 2 * (index ** 2)
    #If no solutions found, we have found the smallest number
    if ticker == 0:
        print(i)
        break

#.0369 ms
print(datetime.now() - z)