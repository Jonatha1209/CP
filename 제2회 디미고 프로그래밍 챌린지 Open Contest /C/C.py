import math
for i in range(int(input())):
 a=int(input())
 if int(math.isqrt(a))*int(math.isqrt(a))==a:
  rn = int(str(a)[::-1])
  rs = int(math.isqrt(rn))
  if rs*rs==rn:print("YES")
  else:print("NO")
 else :
    print("NO")
