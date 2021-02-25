def MSE(x, y):
    return np.mean((x - y) ** 2)

def accuracy(y_pred, y):
    return np.sum(y.T==y_pred)/y.shape[0]
