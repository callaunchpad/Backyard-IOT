from datetime import datetime
from math import floor,ceil
z = datetime.now()

def tentobinary(n):
    index = 0
    list = []
    while n//(2**index) > 1:
        index += 1
    while index >= 0:
        x = int(2**index)
        list.append(n//x)
        if n//x > 0:
            n = n - x
        index -= 1
    return list

def is_palindrome(list):
    list1 = list[:floor(len(list)/2)]
    list2 = list[ceil(len(list)/2):]
    if list1 == list2[::-1]:
        return True
    else:
        return False

sum = 0
for i in range(1,1000000,2):
    list = [int(x) for x in str(i)]
    if is_palindrome(list):
        if is_palindrome(tentobinary(i)):
            sum += i
            print(i)

print(sum)

print(datetime.now() - z)