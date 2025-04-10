import sys
from sys import setrecursionlimit
from collections import defaultdict,deque

class INPUT:
    def __init__(self):
        self.read_buf = sys.stdin.read(1 << 20)
        self.read_idx = 0
        self.__END_FLAG__ = False
        self.__GETLINE_FLAG__ = False

    def __bool__(self):
        return not self.__END_FLAG__

    def IsBlank(self, c):
        return c in ' \n'

    def IsEnd(self, c):
        return c == '\0'

    def _ReadChar(self):
        if self.read_idx == len(self.read_buf):
            self.read_buf = sys.stdin.read(1 << 20)
            if not self.read_buf:
                return '\0'
            self.read_idx = 0
        ret = self.read_buf[self.read_idx]
        self.read_idx += 1
        return ret

    def ReadChar(self):
        ret = self._ReadChar()
        while self.IsBlank(ret):
            ret = self._ReadChar()
        return ret

    def ReadInt(self):
        ret = 0
        cur = self._ReadChar()
        while self.IsBlank(cur):
            cur = self._ReadChar()
        flag = False
        if cur == '-':
            flag = True
            cur = self._ReadChar()
        while not self.IsBlank(cur) and not self.IsEnd(cur):
            ret = 10 * ret + (ord(cur) & 15)
            cur = self._ReadChar()
        if self.IsEnd(cur):
            self.__END_FLAG__ = True
        return -ret if flag else ret

    def ReadString(self):
        ret = []
        cur = self._ReadChar()
        while self.IsBlank(cur):
            cur = self._ReadChar()
        while not self.IsBlank(cur) and not self.IsEnd(cur):
            ret.append(cur)
            cur = self._ReadChar()
        if self.IsEnd(cur):
            self.__END_FLAG__ = True
        return ''.join(ret)

# Macros
#Geometry Macros

def p2v(a, b):
    return (b[0] - a[0], b[1] - a[1])

def ccw(v1, v2):
    res = v1[0] * v2[1] - v1[1] * v2[0]
    
    if res > 0:
        return 1
    elif res < 0:
        return -1
    else:
        return 0
    
def ccw3d(p1, p2, p3):
    res = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    if res > 0:
        return 1
    elif res < 0:
        return -1
    else:
        return 0
    
def isInside_nonconvex(CH, point):
    cnt = 0
    for i in range(len(CH)):
        p1 = CH[i]
        p2 = CH[(i + 1) % len(CH)]
        
        if p1[1] < p2[1]:
            p1, p2 = p2, p1
        
        v1 = p2v(p1, point)
        v2 = p2v(point, p2)
        
        if ccw(v1, v2) == 0:
            if min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]):
                return True
        
        if max(p1[0], p2[0]) < point[0]:
            continue
        if p1[1] <= point[1]:
            continue  # 1
        if p2[1] > point[1]:  # 2
            continue
        if min(p1[0], p2[0]) > point[0]:
            cnt += 1
        elif ccw(v1, v2) > 0:
            cnt += 1
    
    return cnt % 2 != 0

def isInside_convex(CH, point):
    if len(CH) == 0:
        return False

    if len(CH) == 1:
        return CH[0] == point

    if len(CH) == 2:
        return min(CH[0][0], CH[1][0]) <= point[0] <= max(CH[0][0], CH[1][0]) and \
               min(CH[0][1], CH[1][1]) <= point[1] <= max(CH[0][1], CH[1][1])

    l = 0
    r = len(CH) - 1
    while l < r:
        mid = (l + r) // 2
        if ccw3d(CH[0], CH[mid], point) < 0:
            r = mid
        else:
            l = mid + 1

    if l == 0:
        return ccw3d(CH[0], CH[1], point) >= 0 and ccw3d(CH[1], CH[0], point) >= 0

    return ccw3d(CH[0], CH[l - 1], point) >= 0 and \
           ccw3d(CH[l - 1], CH[l], point) >= 0 and \
           ccw3d(CH[l], CH[0], point) >= 0




def monotone_chain(points):
    points = sorted(points)

    lower = []
    for p in points:
        while len(lower) >= 2 and ccw3d(lower[-2], lower[-1], p) <= 0:
            lower.pop() 
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and ccw3d(upper[-2], upper[-1], p) <= 0:
            upper.pop()  
        upper.append(p)

    return lower[:-1] + upper[:-1] 

#others

MAX = 101010101
INF = float("inf")
MINUS_INF = float('-inf')
cin = INPUT()

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def mpow(a, n, m):
    if n == 0:
        return 1
    res = mpow(a, n // 2, m)
    res = (res * res) % m
    if n % 2:
        res = (res * a) % m
    return res

def ncr(n, r, m): 
    if n < r:
        return 0
    res = 1
    for i in range(1, n + 1):
        res = (res * i) % m
    for i in range(1, r + 1):
        res = (res * mpow(i, m - 2, m)) % m
    for i in range(1, n - r + 1):
        res = (res * mpow(i, m - 2, m)) % m
    return res

def lucas(n, r, m):
    if r < 0:
        return 0
    N, R = [], []
    while n or r:
        N.append(n % m)
        R.append(r % m)
        n //= m
        r //= m
    
    res = 1
    for i in range(len(N)):
        res = (res * ncr(N[i], R[i], m)) % m
    return res

def euler_phi(n):
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result

def fibonacci(n):
    if n < 0:
        return None
    seq = {0: 0, 1: 1}
    s = [(n, n)]
    while s[-1][0]:
        s.append(((s[-1][0] - 1) // 2, (s[-1][1] + 1) // 2))
    for a, b in s[::-1]:
        for k in range(a, b + 1):
            if k not in seq:
                seq[k] = (seq[k // 2] ** 2 + seq[k // 2 + 1] ** 2) if k & 1 else (seq[k // 2] ** 2 + 2 * seq[k // 2] * seq[k // 2 - 1])
    return seq[n]

def fibonacci_seq(n):
    if n < 0:
        return None
    seq = [0] * (n + 1) 
    seq[0], seq[1] = 0, 1
    
    for i in range(2, n + 1):
        seq[i] = (seq[i - 1] + seq[i - 2]) % (int(1e9) + 7)  # ✅ 모듈러 적용
    
    return seq


class BerlekampMassey:
    def __init__(self, MOD):
        self.MOD = MOD

    def _pow(self, x, p):
        ret = 1
        piv = x
        while p:
            if p & 1:
                ret = ret * piv % self.MOD
            piv = piv * piv % self.MOD
            p >>= 1
        return ret

    def berlekamp_massey(self, x):
        ls, cur = [], []
        lf, ld = -1, 0
        for i in range(len(x)):
            t = 0
            for j in range(len(cur)):
                t = (t + x[i - j - 1] * cur[j]) % self.MOD
            if (t - x[i]) % self.MOD == 0:
                continue
            if not cur:
                cur = [0] * (i + 1)
                lf, ld = i, (t - x[i]) % self.MOD
                continue
            k = -(x[i] - t) * self._pow(ld, self.MOD - 2) % self.MOD
            c = [0] * (i - lf - 1)
            c.append(k)
            for j in ls:
                c.append(-j * k % self.MOD)
            if len(c) < len(cur):
                c.extend([0] * (len(cur) - len(c)))
            for j in range(len(cur)):
                c[j] = (c[j] + cur[j]) % self.MOD
            if i - lf + len(ls) >= len(cur):
                ls, lf, ld = cur, i, (t - x[i]) % self.MOD
            cur = c
        return [(i % self.MOD + self.MOD) % self.MOD for i in cur]

    def get_nth(self, rec, dp, n):
        m = len(rec)
        s = [0] * m
        t = [0] * m
        s[0] = 1
        if m != 1:
            t[1] = 1
        else:
            t[0] = rec[0]
        
        def mul(v, w):
            m = len(v)
            t = [0] * (2 * m)
            for j in range(m):
                for k in range(m):
                    t[j + k] += v[j] * w[k] % self.MOD
                    if t[j + k] >= self.MOD:
                        t[j + k] -= self.MOD
            for j in range(2 * m - 1, m - 1, -1):
                for k in range(1, m + 1):
                    t[j - k] += t[j] * rec[k - 1] % self.MOD
                    if t[j - k] >= self.MOD:
                        t[j - k] -= self.MOD
            t = t[:m]
            return t

        while n:
            if n & 1:
                s = mul(s, t)
            t = mul(t, t)
            n >>= 1
        
        ret = 0
        for i in range(m):
            ret += s[i] * dp[i] % self.MOD
        return ret % self.MOD

    def guess_nth_term(self, x, n):
        if n < len(x):
            return x[n]
        v = self.berlekamp_massey(x)
        if not v:
            return 0
        return self.get_nth(v, x, n)

class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def __repr__(self):
        return f"Pair({self.first}, {self.second})"

    def __eq__(self, other):
        if isinstance(other, Pair):
            return self.first == other.first and self.second == other.second
        return False

    def __add__(self, other):
        if isinstance(other, Pair):
            return Pair(self.first + other.first, self.second + other.second)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Pair):
            return Pair(self.first - other.first, self.second - other.second)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Pair):
            return Pair(self.first * other.first, self.second * other.second)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Pair):
            return Pair(self.first / other.first, self.second / other.second)
        return NotImplemented

class Primetest:
   
    def PrimeTest(self, a: int, n: int, s: int) -> bool:
        
        if a >= n:
            a %= n

        if a < 2:
            return True
        
        d = n >> s

        x = pow(a, d, n)

        if x == n - 1:
            return True
        if x == 1:
            return True
        
        for _ in range(s):
            x = pow(x, 2, n)
            if x == 1:
                return False
            
            if x == n - 1:
                return True

        return False
    
    def process(self, n: int) -> bool:
       
        if n == 2:
            return True
        
        if n < 2 or ~n & 1:
            return False
        
        r, d = 1, n >> 1
        while ~d & 1:
            d >>= 1  
            r += 1  

        return all(self.PrimeTest(i, n, r) for i in ([2, 7, 61] if n < 4759123141 else [2, 325, 9375, 28178, 450775, 9780504, 1795265022]))
import random
randrange=random.randrange
class Factorize:
    def _f(self, x: int, a: int, n: int) -> int:
        return (pow(x, 2) + a) % n

    def __pollard_rho(self, n: int) -> int:
        x = randrange(1, n)
        c = randrange(1, n)
        y, g = x, 1

        while g == 1:
            x = self._f(x, c, n)
            y = self._f(self._f(y, c, n), c, n)
            g = gcd(abs(x - y), n)
        if g == n:
            return self.__pollard_rho(n)
        return g

    def factorize(self, n: int) -> list[int]:
        if n == 1:
            return []
        if ~n & 1:
            return [2] + self.factorize(n >> 1)
        if Primetest().process(n):
            return [n]
        f = self.__pollard_rho(n)
        return self.factorize(f) + self.factorize(n // f)

readint = cin.ReadInt
readchar = cin.ReadChar
readstring = cin.ReadString

#main


a=readint()
print(a*(a+1)//2)
