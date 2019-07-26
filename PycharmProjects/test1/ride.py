from datetime import datetime
z = datetime.now()

def convert(list):
    number = sum(d*10**i for i,d in enumerate(list[::-1]))
    return number

lina = []

for denominator in range(11,100):
    for numerator in range(10,denominator):
        fraction = numerator/denominator
        list1 = [int(i) for i in str(numerator)]
        list2 = [int(i) for i in str(denominator)]
        if set(list1) & set(list2) and convert(list1)%10 != 0 and convert(list2)%10 != 0:
            x = list(set(list1) & set(list2))
            list1.remove(x[0])
            list2.remove(x[0])
            if convert(list2) != 0:
                if convert(list1)/convert(list2) == fraction:
                    lina.append(convert(list1))
                    lina.append(convert(list2))

mult = 1
for i in range(0,6):
    mult *= lina[i]

print(mult)

print(lina)
print(datetime.now() - z)
