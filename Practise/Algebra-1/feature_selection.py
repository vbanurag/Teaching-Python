# %%
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# %%
def log(message, exp):
    print(message,exp)
# %%
## feature selection
def select_independent_feature(X, threshold= 0.5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _, s, _ = np.linalg.svd(X_scaled)
    cum_variance = np.cumsum(s**2)/np.sum(s**2)
    r = np.argmax(cum_variance >= threshold) + 1
    return r

X,Y = load_iris(return_X_y=True)
num_feature = select_independent_feature(X)
log("X in Iris: \n", X)
print ( f" Number of linearly independent features :{ num_feature }")
print ( f" Original number of features : {X. shape[1]} ")
# %%
