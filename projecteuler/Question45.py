from datetime import datetime
import math
z = datetime.now()

#Detecting if a number is a perfect square
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

#Detecting if a number is Triangular
def is_tri(n):
    if is_square(8*n+1):
        return True
    return False

#Detecting if a number is Pentagonal
def is_penta(n):
    if is_square(24*n+1):
        zee = math.sqrt(24*n+1)
        zee = (zee + 1)/6
        if zee == int(zee):
            return True
    return False

#Iteration through hexagonal numbers until a number is found that satisfies all 3 criteria
#Starts at 143 (lowest number that is triangular, pentagonal, and hexagonal
for i in range(144,1000000):
    num = i*(2*i-1)
    if is_penta(num):
        if is_tri(num):
            print(num,i)
            break


print(datetime.now() - z)