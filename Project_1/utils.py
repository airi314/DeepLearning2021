def scale(X):
    return (X-np.mean(X, 0))/np.std(X, 0)

def one_hot_encode(y):
    y = pd.DataFrame(y)
    y = pd.get_dummies(y[0])
    return np.array(y)

def shuffle_batch(x, y):
    t = np.arange(x.shape[0])
    np.random.shuffle(t)
    return x[t], y[t]
