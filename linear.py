import numpy as np

import matplotlib.pyplot as plt

data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
data_y = np.sin(data_x) + 0.1*np.power(data_x,2) + 0.5*np.random.randn(100,1)
data_x /= np.max(data_x)

print(data_y)
data_x = np.hstack((np.ones_like(data_x), data_x))

order = np.random.permutation(len(data_x))
portion = 20

test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]

print(test_x)

plt.plot(train_x, train_y, 'o')

def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, np.sum(np.power(error, 2))

w = np.random.randn(2)
alpha = 0.5
tolerance = 1e-5

iteration = 1
while True:
    gradient, error = get_gradient(w,train_x,train_y)
    new_w = w - alpha*gradient
    if np.sum(abs(new_w-w)) < tolerance:
        print("converged")
        break
    if iteration%10 == 0 or iteration < 10:
        print("Iteration: %d, Error: %.4f" %(iteration, error))
        print(w[0])
        plt.plot(train_x, train_y, 'o')
        line_x = np.linspace(0,1,10)
        line_y = w[0] + w[1]*line_x
        plt.plot(line_x, line_y)
        plt.show()
    iteration += 1
    w = new_w
