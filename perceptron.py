import numpy as np
import matplotlib.pyplot as plt
from data import big_data, big_data_labels, gen_lin_separable, gen_flipped_lin_separable, big_higher_dim_separable

def poly2d_features(X, order: int) -> np.ndarray:

    if order < 1:
        raise ValueError("order must be >= 1")

    x1 = X[:, 0]
    x2 = X[:, 1]
    feats = []
    for deg in range(order + 1):
        for i in range(deg + 1):
            j = deg - i
            feats.append((x1 ** i) * (x2 ** j))
    return np.column_stack(feats)

def scatter_plot(ax, x, y, weights, bias, order: int, epoch):
    ax.cla()

    # mesh grid over plot area in original x-space
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )
    grid_xy = np.c_[xx.ravel(), yy.ravel()]

    # evaluate classifier on transformed grid
    Phi_grid = poly2d_features(grid_xy, order)
    scores = Phi_grid @ weights + bias

    Z = np.sign(scores)
    Z[Z == 0] = 1
    Z = Z.reshape(xx.shape)

    # decision regions
    ax.contourf(xx, yy, Z, alpha=0.2, levels=[-1, 0, 1], colors=["red", "blue"])

    # boundary where score == 0 (solid)
    ax.contour(xx, yy, scores.reshape(xx.shape), levels=[0], linewidths=2, colors=["black"])

    # points
    ax.scatter(x[y == -1, 0], x[y == -1, 1], c="red",  label="class red",  s=100)
    ax.scatter(x[y ==  1, 0], x[y ==  1, 1], c="blue", label="class blue", s=100)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"(order={order}), (epoch={epoch})")
    ax.legend(loc="best")

def perceptron(x, y, lr, iterations, order: int = 1, shuffle: bool = True):


    x = x.T
    y = y.flatten()


    Phi = poly2d_features(x, order)

    bias = 0.0
    n_samples, n_features = Phi.shape
    weights = np.zeros(n_features)

    plt.ion()
    fig = plt.figure(1, figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(111)
    fig.show()

    for epoch in range(iterations):
        idx = np.random.permutation(n_samples) if shuffle else np.arange(n_samples)

        for i in idx:
            if y[i] * (Phi[i] @ weights + bias) <= 0:
                weights += lr * y[i] * Phi[i]
                bias += lr * y[i]

        scatter_plot(ax, x, y, weights, bias, order, epoch)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.05)

    plt.ioff()
    plt.show()
    return weights, bias

if __name__ == "__main__":
    x, y = big_higher_dim_separable() # You can use any data-generating function from data.py
    weights, bias = perceptron(x, y, lr=0.0001, iterations=1000, order=15) # Define learning ate, iterations and order
