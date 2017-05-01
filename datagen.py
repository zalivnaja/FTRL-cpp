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
    w_num_size = 10
    samples_count = 1000
    bernoulli_count = 10
    
    categorical_size = 5
    categorical_count = 10
    
    w_size = w_num_size + categorical_size * categorical_count
    
    columns = ["id"] + ['var_' + str(i) for i in range(w_num_size + categorical_count)] + ["target"]
    with open(filename, 'wt') as testfile:
        testfile.write(','.join(columns) + '\n') 
    
        w = np.random.rand(w_size, 1)
        b = np.random.rand(1, 1)[0][0] - 3

        ind = 0
        for sample_i in range(samples_count):
            x = np.random.rand(1, w_num_size)
            
            cat_x = np.random.choice(categorical_size, categorical_count)
            cat_x = np.reshape(cat_x, (1, categorical_count))
            
            dummy_cat_x = np.zeros((1, categorical_size * categorical_count))
            for cat_i in range(categorical_count):
                dummy_cat_x[0, cat_i * categorical_size + cat_x[0, cat_i]] = 1
            
            full_dummy_x = np.concatenate([x, dummy_cat_x], axis = 1)
            full_x = np.concatenate([x, cat_x], axis = 1)
            
            p = sigmoid(full_dummy_x.dot(w) + b)
            ys = bernoulli.rvs(p, size=bernoulli_count)
            
            str_x = ', '.join(["{0:.2f}".format(x_i) for x_i in x[0]]) + ', ' + ', '.join([str(x_i) for x_i in cat_x[0]])
            # str_x = np.array2string(full_x[0], precision=2, separator=',', suppress_small=True)

            # print(str_x)

            for y in ys:
                testfile.write(str(ind) + ',' + str_x[1:-1] + ',' + str(y) + '\n') 
                ind += 1
        
generate_simple_test('categ_testdata.csv')