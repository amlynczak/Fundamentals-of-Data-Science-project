from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def reduction(train_data):
    scaler = StandardScaler()

    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)
    data_red_2d = pca_2d.fit_transform(train_data)
    data_red_3d = pca_3d.fit_transform(train_data)

    return data_red_2d, data_red_3d