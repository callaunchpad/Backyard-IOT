pd=1
pn=1
fracs=[]
for n in range(10,100):
    for d in range(n+1,100):
        ns=str(n)
        ds=str(d)
        if '0' not in ns+ds:
            if ns[1]==ds[0] and int(ns[0])/int(ds[1])==n/d:
                pn=pn*n
                pd=pd*d
                fracs.append(str(n)+"/"+str(d))
                break
i=2
while i <pd:
    while pn%i==0 and pd%i==0:
        pn=pn/i
        pd=pd/i
    i=i+1
print(pd,fracs)