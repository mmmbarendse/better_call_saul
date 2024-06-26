def gamma(alpha, beta, N, plot=False):
    import random
    import matplotlib.pyplot as plt

    wealtharr = [random.gammavariate(alpha, beta) for  in range(N)]
    sum_wealth = sum(wealth_arr)
    wealth_arr = [i/sum_wealth for i in wealth_arr]    # normalization

    if plot:
        plt.hist(wealth_arr, bins=100)
        plt.show()
    return wealth_arr

def gini(arr, plot=False):
    import numpy as np
    import matplotlib.pyplot as plt

    X_lorenz = np.cumsum(np.sort(arr)) / np.sum(arr)
    X_lorenz = np.insert(X_lorenz, 0, 0)

    # Compute the Gini coefficient
    B = np.trapz(X_lorenz, dx=1/(X_lorenz.size - 1))
    A = 0.5 - B
    Gini_coefficient = A / 0.5

    # Plotting the Lorenz curve
    if plot:
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.scatter(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz, color='darkgreen', s=10)
        ax.plot([0, 1], [0, 1], color='k')
        ax.set_title('Lorenz curve')
        plt.show()

    return Gini_coefficient