from sklearn.decomposition import PCA

def pca(x):
    fit = PCA(n_components=x.shape[1]).fit(x)
    return fit.components_.T