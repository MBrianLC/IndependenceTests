# Chi-Square (modified)

from scipy.stats import chisquare

# 2^m x 2^n matrix that takes the occurrences of m bits after n bits (without overlapping)
def Ocurrence_01(u,m,n):
    s = n+m
    auxn = 2**(n-1)
    auxm = 2**(m-1)
    mat = [[0 for j in range(auxn*2)] for i in range(auxm*2)]               
    ant = int(u[:n],2)
    act = int(u[n:n+m],2)
    mat[act][ant] += 1
    c = s
    while (c + s < len(u)):
        ant = int(u[c:n+c],2)
        act = int(u[n+c:n+m+c],2)
        mat[act][ant] += 1
        c += s
    return mat

# Chi-Square test
def chi_square(u,m,n):
    mat = Ocurrence_01(u,m,n)
    sp = chisquare(np.array(mat).ravel())
    return (sp[0], sp[1])

# Given i with 1 <= i <= total, it separates the array into blocks of i bytes
def matBlocks(u,total):
    mat = []
    l = len(u)
    for i in range(2,total+1):
        b = l//i
        r = b*i
        mat.append([int(u[start:start+i],2) for start in range(0, r, i)])
    return mat

# Given mat_b from matBlocks, it creates the 2^m x 2^n matrix
def sep(m,n,mat_b):
    auxn = 2**n
    auxm = 2**m
    mat = np.zeros(auxn*auxm)
    f = collections.Counter(mat_b)
    for i in range(auxn*auxm):
        mat[i] = f[i]
    return mat

# Executes the Chi-Square test for i, j <= total
def total_chi_square(u,total):
    sp = []
    mat = matBlocks(u,2*total)
    for i in range (total):
        for j in range (i,total):
            oc_mat = sep(i+1,j+1,mat[i+j])
            sp.append(chisquare(np.array(oc_mat)))
    return sp
    
# Fisher's exact test 
    
# rpy2: Laurent Gautier. GNU General Public License v2 or later (GPLv2+) (GPLv2+)
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
rpy2.robjects.numpy2ri.activate()

rstats = importr('stats')
def fisher(u,m,n):
    mat = Ocurrence_01(u,m,n)
    return rstats.fisher_test(np.array(mat))[0][0]   

# Hamming tests

from scipy.stats import binom

# Hamming weight: divides the sequence into blocks (without overlapping) and counts the number of 1s on each ones
def Hamming_weight_consec(u,n):
    size = 2**n
    aux = len(u)//size
    ones = []
    for i in range(aux):
        f = collections.Counter(u[size*i:size*(i+1)])
        ones.append(f['1'])
    return ones  

# Hamming distance: divides the sequence into blocks (without overlapping) and calculates the hamming distance every two blocks
def Hamming_distance_consec(u,n):
    size = 2**n
    aux = len(u)//size
    ones = []
    act = int(u[:size],2)
    for i in range(aux-1):
        nxt = int(u[size*(i+1):size*(i+2)],2)
        dif = bin(act^nxt)[2:].zfill(size)
        f = collections.Counter(dif)
        act = nxt
        ones.append(f['1'])
    return ones    
    
# Calculates Hamming weight or distance tests (by default in Hamming weight)
def Hamming_test(u,n,weight=True):
    ones = []
    if weight:
        ones = Hamming_weight_consec(u,n)
    else:
        ones = Hamming_distance_consec(u,n)
    x = binom(2**n,0.5)
    l1 = []
    l2= []
    l = len(ones)
    f = collections.Counter(ones)
    r = 0
    suma = 0
    while x.pmf(r)*l < 5:
        suma += f[r]
        r += 1
    l1.append(x.cdf(r-1)*l)
    l2.append(suma)
    for i in range(r,2**n+1-r):
        l1.append(x.pmf(i)*l)
        l2.append(f[i])
        suma += f[i]
    l1.append(x.cdf(r-1)*l)
    l2.append(l-suma)
    sp = chisquare(l2,l1)
    return (sp[0], sp[1])    
    
# Optimized version, which calculates both the Hamming weight and the Hamming distance test
def Hamming_test_wd(u,n):
    s = 2**n
    l = len(u)
    b = l//s
    r = b*s
    ones_w, ones_d = [], []
    mat = [u[start:start+s] for start in range(0, r, s)]
    a = mat[0]
    act = int(a,2)
    f = collections.Counter(a)
    ones_w.append(f['1'])
    for i in range(1,b):
        a = mat[i]
        nxt = int(a,2)
        dif = bin(act^nxt)[2:].zfill(s)
        act = nxt
        f = collections.Counter(a)
        ones_w.append(f['1'])
        f = collections.Counter(dif)
        ones_d.append(f['1'])
    x = binom(s,0.5)
    l1,l2,l3,l4 = [],[],[],[]
    l = len(ones_w)
    f1,f2 = collections.Counter(ones_w), collections.Counter(ones_d)
    r1 = 0
    suma1,suma2 = 0,0
    while x.pmf(r1)*l < 5:
        suma1 += f1[r1]
        suma2 += f2[r1]
        r1 += 1
    r2 = r1
    while x.pmf(r2)*(l-1) < 5:
        suma2 += f2[r2]
        r2 += 1
    l1.append(x.cdf(r1-1)*l)
    l2.append(suma1)
    l3.append(x.cdf(r2-1)*(l-1))
    l4.append(suma2)
    for i in range(r1,r2):
        l1.append(x.pmf(i)*l)
        l2.append(f1[i])
        suma1 += f1[i]
    for i in range(r2,s+1-r2):
        l1.append(x.pmf(i)*l)
        l2.append(f1[i])
        l3.append(x.pmf(i)*(l-1))
        l4.append(f2[i])
        suma1 += f1[i]
        suma2 += f2[i]
    for i in range(s+1-r2,s+1-r1):
        l1.append(x.pmf(i)*l)
        l2.append(f1[i])
        suma1 += f1[i]
    l1.append(x.cdf(r1-1)*l)
    l2.append(l-suma1)
    l3.append(x.cdf(r2-1)*(l-1))
    l4.append(l-1-suma2)
    spw = chisquare(l2,l1)
    spd = chisquare(l4,l3)
    return (spw[0], spw[1]), (spd[0], spd[1])

# Runs all independence tests on a sequence u 
def iTests(u,n,m):
    
    # Chi-square
    sp = total_chi_square(u,n)
    
    # Fisher
    sp += [(0,fisher(u,1,1))]
    
    # Hamming
    for i in range (5,5+m):
        sp += Hamming_test_wd(u,i)
    return sp

# Runs all independence tests on each sequence of u (using iTests)    
def itests(u,n,m):
    l = len(u)
    s = [[] for i in range(l)]
    p = [[] for i in range(l)]
    for i in range(len(u)):
        (s[i], p[i]) = zip(*iTests(u[i],n,m))
    return (s,p)