# %% 
import numpy as np
from sklearn.preprocessing import StandardScaler

# %%
def log(message, exp):
    print(message,exp)
# %%
#Sample dataset : House features (area , bedrooms, age)
houses = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 3, 6]
])

#standarisez the feature
scaler = StandardScaler()
houses_scaled = scaler.fit_transform(houses)

# %% 
print("Original Houses: \n", houses)
print("Scaled Houses: \n", houses_scaled)
# %%
## Vector Addition in Np
a = np.array([1,3,5])
b = np.array([3,6,8])

print("Sub: \n", abs(a-b))
print("Addition: \n", a+b)
# %%
## Scaler Multiplication in Python
scaler = 2.5

print("Scaler Mult: \n", scaler * a)

# %%
## Dot Product and Cosine Similarity
##Cosine similarity measures the similarity 
# between two vectors of an inner product space. 
# It is measured by the cosine of the angle 
# between two vectors and determines 
# whether two vectors are pointing in roughly the 
# same direction. It is often used to measure document
#  similarity in text analysis.
def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a) * np.linalg.norm(b))

doc1 = np.array([1,1,0,1,0,1])
doc2 = np.array([1,1,1,0,1,0])
log("Cosine Similaity: \n", cosine_similarity(doc1,doc2) )


# %%
## Linear independdence in NP
def is_linearly_inde(vectors):
    matrix = np.array(vectors).T
    rank = np.linalg.matrix_rank(matrix)
    return rank == matrix.shape[1]

# Example usage
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
v4 = np.array([1, 1, 1])

log("Set 1: \n", is_linearly_inde([v1,v2,v3]))
log("Set 2: \n", is_linearly_inde([v1,v2,v3,v4]))
# %%
## Norms in Python

def l1_norm(x):
    return np.sum(np.abs(x))

def l2_norm(x):
    return np.sqrt(np.sum(x**2))

def lp_norm(x, p):
    return np.sum(np.abs(x)**p) ** (1/p)

x = np.array([1, -2, 3, -4, 5])

log("L1 Norm: \n",l1_norm(x))
log("L1 in NP: \n", np.linalg.norm(x, ord=1))

log("L2 Norm: \n",l2_norm(x))
log("L2 in NP: \n", np.linalg.norm(x, ord=2))


log("LP Norm: \n",lp_norm(x, 3))
log("LP in NP: \n", np.linalg.norm(x, ord=3))


# %%
