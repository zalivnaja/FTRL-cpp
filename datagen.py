import numpy as np
from scipy.stats import bernoulli

np.random.seed(4242)

def sigmoid(x, eps = 35.0):
    x = max(min(x, eps), -eps)

    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def generate_simple_test(filename):
    w_size = 10
    samples_count = 100
    bernoulli_count = 10
    
    columns = ["id"] + ['var_' + str(i) for i in range(w_size)] + ["target"]
    with open(filename, 'wt') as testfile:
        testfile.write(','.join(columns) + '\n') 
    
        w = np.random.rand(w_size, 1)
        b = np.random.rand(1, 1)[0][0] - 3

        ind = 0
        for i in range(samples_count):
            x = np.random.rand(1, w_size)
            p = sigmoid(x.dot(w) + b)
            ys = bernoulli.rvs(p, size=bernoulli_count)
            
            str_x = np.array2string(x[0], precision=2, separator=',', suppress_small=True)
            for y in ys:
                testfile.write(str(ind) + ',' + str_x[1:-1] + ',' + str(y) + '\n') 
                ind += 1
        
generate_simple_test('testdata.csv')