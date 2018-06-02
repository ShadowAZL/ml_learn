import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


def fitting():
    data = pd.read_csv('house_data.txt', names=['area', 'bedroom', 'price'])

    x_data = data[['area', 'bedroom']]
    x_data = (x_data - x_data.mean())/(x_data.max() - x_data.min())
    y_data = data['price']
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(x_data['area'], y_data, 'rx')
    plt.title('area-price')

    plt.subplot(2, 2, 2)
    plt.plot(x_data['bedroom'], y_data, 'bx')
    plt.title('bedroom-price')


    alpha = 10
    max_iter = 50

    model = LinearRegression(alpha, max_iter)
    loss, _ = model.fit(x_data.values, y_data.values)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, max_iter+1), loss)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

fitting()

