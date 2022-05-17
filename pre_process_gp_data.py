import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA,KernelPCA


if __name__ == '__main__':
    fn = 'household'

    if not os.path.exists(fn):
        os.makedirs(fn)
    df = pd.read_csv('household_power_consumption.txt', sep=';',low_memory=False)
    df['Time'] =df['Time'].apply(lambda x: float(x[:2]))
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True).dt.month
    df = df.dropna()
    y = df['Global_active_power'].values.astype('float')
    X = df.drop(['Global_active_power'],axis=1).values.astype('float')
    pca = PCA(n_components=3, svd_solver='full')
    # pca = KernelPCA(n_components=3,kernel='rbf')
    pca.fit(X)
    print(np.sum(pca.explained_variance_ratio_))

    with open(f'{fn}/X.npy', 'wb') as f:
        np.save(f, X)
    with open(f'{fn}/y.npy', 'wb') as f:
        np.save(f, y)


    fn = '3D_spatial'

    if not os.path.exists(fn):
        os.makedirs(fn)
    df = pd.read_csv('3D_spatial_network.csv')
    df = df.dropna()
    X = df[['X2','X3']].values
    y = df['X4'].values
    with open(f'{fn}/X.npy', 'wb') as f:
        np.save(f, X)
    with open(f'{fn}/y.npy', 'wb') as f:
        np.save(f, y)


    fn = 'Song'

    if not os.path.exists(fn):
        os.makedirs(fn)
    df = pd.read_csv('YearPredictionMSD.txt',header=None)
    df = df.dropna()
    y = df.values[:,0]
    X = df.values[:,1:]
    pca = PCA(n_components=3, svd_solver='full')
    # pca = KernelPCA(n_components=3,kernel='rbf')
    pca.fit(X)
    print(np.sum(pca.explained_variance_ratio_))
    with open(f'{fn}/X.npy', 'wb') as f:
        np.save(f, X)
    with open(f'{fn}/y.npy', 'wb') as f:
        np.save(f, y)


