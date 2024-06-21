def gamma(alpha, beta, N):
    import random
    import matplotlib.pyplot as plt

    wealth_arr = [random.gammavariate(alpha, beta) for _ in range(N)]
    plt.hist(wealth_arr, bins=100)
    plt.show()
    return wealth_arr

def gini(arr):
    import numpy as np
    import matplotlib.pyplot as plt

    X_lorenz = np.cumsum(np.sort(arr)) / np.sum(arr)
    X_lorenz = np.insert(X_lorenz, 0, 0)

    # Compute the Gini coefficient
    B = np.trapz(X_lorenz, dx=1/(X_lorenz.size - 1))
    A = 0.5 - B
    Gini_coefficient = A / 0.5

    # Plotting the Lorenz curve
    fig, ax = plt.subplots(figsize=[6, 6])
    ax.scatter(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, color='darkgreen', s=10)
    ax.plot([0, 1], [0, 1], color='k')
    ax.set_title('Lorenz curve')
    plt.show()

    return Gini_coefficient