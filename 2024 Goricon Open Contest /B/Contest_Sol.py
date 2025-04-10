def is_prime(x):
    if x<2:
        return False
    for i in range(2,int(x**0.5)+1):
        if x%i==0:
            return False
    return True

limit = 2000000
a,b=map(int,input().split())

l=[]
for i in range(1,limit+1):
    if not is_prime(i):
        l.append(i)
res=[]
for i in range(len(l)):
    seq=[l[i]+j*b for j in range(a)]
    if all(not is_prime(num)for num in seq):
        res=seq
        break 
if res:
    print(" ".join(map(str,res)))
else:
    print(-1)
