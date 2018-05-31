import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


def muti_feature():
    data = pd.read_csv('house_data.txt', names=['area', 'bedroom', 'price'])

    x_data = data[['area', 'bedroom']]
    x_data = (x_data - x_data.mean())/(x_data.max() - x_data.min())
    y_data = data['price']
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 2, 1)
    plt.plot(x_data['area'], y_data, 'rx')
    plt.title('area-price')

    plt.subplot(3, 2, 2)
    plt.plot(x_data['bedroom'], y_data, 'bx')
    plt.title('bedroom-price')


    alpha = 0.01
    max_iter = 200

    model = LinearRegression(alpha, max_iter)
    loss = model.fit(x_data.values, y_data.values)
    plt.subplot(3, 1, 2)
    plt.plot(np.arange(1, max_iter+1), loss)

    plt.subplots_adjust(hspace=0.4)
    plt.show()


def single():
    data = pd.read_csv('single_house_data.txt', names=['area', 'price'])
    x_data = data['area']
    y_data = data['price']

    plt.subplot(2, 1, 1)
    plt.plot(x_data, y_data, 'rx')
    alpha = 0.01
    max_iter = 1500
    model = LinearRegression(alpha, max_iter)
    loss = model.fit(x_data, y_data)
    pre_data = model.predict(x_data)
    plt.plot(x_data, pre_data, 'b-')

    plt.subplot(2, 1, 2)
    plt.title('Loss Curve')
    plt.grid(True)
    plt.plot(np.arange(1, max_iter+1), loss, 'go')

    plt.show()


def single_3d():
    data = pd.read_csv('single_house_data.txt', names=['area', 'price'])
    x_data = data['area']
    y_data = data['price']

    alpha = 0.01
    max_iter = 1500
    model = LinearRegression(alpha, max_iter)
    loss = model.fit(x_data, y_data)
    pre_data = model.predict(x_data)

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    tmp_model = LinearRegression(0, 0)
    theta_0 = np.arange(-10, 10, .5)
    theta_1 = np.arange(-1, 4, .1)

    x = []
    y = []
    cost = []

    for i in theta_0:
        for j in theta_1:
            tmp_model.w = np.array([i, j])
            x.append(i)
            y.append(j)
            cost.append(np.mean((tmp_model.predict(x_data) - y_data)**2))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, cost, 'bo-', c=np.abs(cost), cmap=plt.get_cmap('YlOrRd'))

    w_list = np.array(model.w_list)
    plt.plot(w_list[:, 0], w_list[:, 1], loss, 'bo-')

    plt.show()

single()
