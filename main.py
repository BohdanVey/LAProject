import random
import time
"""
It work only with square matrix
"""

def new_matrix(r, c):
    """Create a new matrix filled with zeros."""
    matrix = [[0 for row in range(r)] for col in range(c)]
    return matrix


def direct_multiply(x, y):
    if len(x[0]) != len(y):
        return "Multiplication is not possible!"
    else:
        p_matrix = new_matrix(len(x), len(y[0]))
        for i in range(len(x)):
            for j in range(len(y[0])):
                for k in range(len(y)):
                    p_matrix[i][j] += x[i][k] * y[k][j]
    return p_matrix


def split(matrix):
    """Split matrix into quarters."""
    a = b = c = d = matrix
    x_max = len(a) - len(a) // 2
    y_max = len(a[0]) - len(a[0]) // 2
    a = a[:x_max]
    b = b[:x_max]
    c = c[x_max:]
    d = d[x_max:]
    for i in range(len(a)):
        a[i] = a[i][:y_max]
    for i in range(len(b)):
        b[i] = b[i][y_max:]
        if len(b[i]) != y_max:
            b[i].append(0)
    for i in range(len(c)):
        c[i] = c[i][:y_max]
    for i in range(len(d)):
        d[i] = d[i][y_max:]
        if len(d[i]) != y_max:
            d[i].append(0)
    if len(c) != x_max:
        c.append([0] * y_max)
        d.append([0] * y_max)
    return a, b, c, d


def add_matrix(a, b):
    if type(a) == int:
        d = a + b
    else:
        d = []
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(a[i][j] + b[i][j])
            d.append(c)
    return d


def subtract_matrix(a, b):
    if type(a) == int:
        d = a - b
    else:
        d = []
        for i in range(len(a)):
            c = []
            for j in range(len(a[0])):
                c.append(a[i][j] - b[i][j])
            d.append(c)
    return d


def shape(x):
    return len(x), len(x[0])


def strassenR(x, y):
    if len(x) < 64:
        return direct_multiply(x, y)
    else:
        a, b, c, d = split(x)
        e, f, g, h = split(y)
        p1 = strassenR(a, subtract_matrix(f, h))

        p2 = strassenR(add_matrix(a, b), h)

        p3 = strassenR(add_matrix(c, d), e)

        p4 = strassenR(d, subtract_matrix(g, e))

        p5 = strassenR(add_matrix(a, d), add_matrix(e, h))

        # p6 = (b-d)*(g+h)
        p6 = strassenR(subtract_matrix(b, d), add_matrix(g, h))

        # p7 = (a-c)*(e+f)
        p7 = strassenR(subtract_matrix(a, c), add_matrix(e, f))
        z11 = add_matrix(subtract_matrix(add_matrix(p5, p4), p2), p6)

        z12 = add_matrix(p1, p2)

        z21 = add_matrix(p3, p4)

        z22 = add_matrix(subtract_matrix(subtract_matrix(p5, p3), p7), p1)

        z = new_matrix(len(x), len(y[0]))
        for i in range(len(z11)):
            for j in range(len(z11)):
                z[i][j] = z11[i][j]
                if j + len(z11) < len(y[0]):
                    z[i][j + len(z11)] = z12[i][j]
                if i + len(z11) < len(x):
                    z[i + len(z11)][j] = z21[i][j]
                if i + len(z11) < len(x) and j + len(z11) < len(y[0]):
                    z[i + len(z11)][j + len(z11)] = z22[i][j]
        return z


def strassen(x, y):
    x_max = len(x)
    y_max = len(y[0])
    c = strassenR(x, y)
    return c


N = 800
M = 800
K = 800
a = []
b = []
for i in range(N):
    a.append([])
    for j in range(M):
        a[-1].append(random.randint(1, 100))

for i in range(M):
    b.append([])
    for j in range(K):
        b[-1].append(random.randint(1, 100))
time_direct_st = time.time()
x = direct_multiply(a, b)
time_direct = time.time() - time_direct_st
time_strassen_st = time.time()
y = strassen(a, b)
time_strassen = time.time() - time_strassen_st
for i in range(len(x)):
    for j in range(len(x[0])):
        if x[i][j] != y[i][j]:
            print("WRONG")
print(f"Time strassen {time_strassen}")
print(f"Time Direct {time_direct}")
