import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from linear_regression import LinearRegression


def plot_data():
    data = pd.read_csv('food_profit.txt', names=['population', 'profit'])
    x = data['population']
    y = data['profit']

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'rx', markersize=10)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.grid(True)
    plt.show()


def fitting():
    data = pd.read_csv('food_profit.txt', names=['population', 'profit'])
    x = data['population']
    y = data['profit']

    alpha = 0.01
    max_iter = 1500
    model = LinearRegression(alpha, max_iter)
    loss, _ = model.fit(x, y)
    p = model.predict(x)

    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(np.arange(1, 1501), loss)
    plt.title('Loss Curve')

    plt.subplot(2,1,2)
    plt.plot(x, y, 'rx', markersize=10, label='Traing Data')
    plt.plot(x, p, 'b', label='Linear Regression')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.grid(True)
    plt.legend()
    plt.show()


def cost_test():
    data = pd.read_csv('food_profit.txt', names=['population', 'profit'])
    x = data['population']
    y = data['profit']

    x = x[:, np.newaxis]
    x = np.hstack((np.ones_like(x), x))
    w = np.zeros(2)
    cost = 0.5 * np.mean((x.dot(w) - y)**2)
    print('Cost is %.2f when theta is zero' % cost)


def visual_contour():
    data = pd.read_csv('food_profit.txt', names=['population', 'profit'])
    x = data['population']
    y = data['profit']

    x = x[:, np.newaxis]
    x = np.hstack((np.ones_like(x), x))

    theta0 = np.zeros((40, 50))
    theta1 = np.zeros((40, 50))
    jvals = np.zeros((40, 50))
    for i, t0 in enumerate(np.arange(-10, 10, 0.5)):
        for j, t1 in enumerate(np.arange(-1, 4, 0.1)):
            theta0[i, j] = t0
            theta1[i, j] = t1
            jvals[i, j] = 0.5*np.mean((x.dot(np.array([t0,t1])) - y)**2)
    plt.figure(figsize=(10, 6))
    cs = plt.contour(theta0, theta1, jvals, 80, extent=[-10, 10, -1, 4])
    plt.plot([-4.2], [1.3], 'rx')
    plt.show()


def visual_3d():
    data = pd.read_csv('food_profit.txt', names=['population', 'profit'])
    x = data['population']
    y = data['profit']

    x = x[:, np.newaxis]
    x = np.hstack((np.ones_like(x), x))

    theta0 = np.zeros((40, 50))
    theta1 = np.zeros((40, 50))
    jvals = np.zeros((40, 50))
    for i, t0 in enumerate(np.arange(-10, 10, .5)):
        for j, t1 in enumerate(np.arange(-1, 4, .5)):
            theta0[i, j] = t0
            theta1[i, j] = t1
            jvals[i, j] = 0.5*np.mean((x.dot(np.array([t0, t1])) - y)**2)

    import mpl_toolkits.mplot3d
    ax = plt.gca(projection='3d')
    ax.plot_surface(theta0, theta1, jvals, cmap=plt.get_cmap('BuPu_r'))

    alpha = 0.01
    max_iter = 1500
    model = LinearRegression(alpha, max_iter)
    loss, w_list = model.fit(x, y)
    w_list = np.array(w_list)

    plt.plot([w_list[0,0]], [w_list[0,1]], [loss[0]], 'rx')
    plt.plot(w_list[:, 0], w_list[:, 1], loss, 'o')
    plt.plot([w_list[-1, 0]], [w_list[-1, 1]], [loss[-1]], 'gx')
    plt.show()

fitting()
