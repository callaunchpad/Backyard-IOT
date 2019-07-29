from datetime import datetime
from math import sqrt
z = datetime.now()

def is_square(apositiveint):
  if apositiveint == 1:
      return True
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def word2num(word):
    sum = 0
    for letter in word:
        if ord(letter) - 64 >= 0:
            sum += ord(letter) - 64
    return sum

file = open('words.txt','r+')
i = ""
for word in file:
    i = i + word
i = i.split(',')

triword = 0
for word in i:
    newnum = word2num(word)
    print(newnum)
    if is_square(8*newnum+1):
        triword += 1

print(triword)









print(datetime.now() - z)