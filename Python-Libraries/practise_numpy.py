# %%
print("hello world!")

# %%
import numpy as np


# %%
a = np.array([1,2,4])
print(a.ndim, ':: Dimension')
print(a.size,':: size')
print(a.shape, ':: Shape')
print(a.nbytes, ':: bytes')
print(a)
# %%

b = np.array([[1,2,4], [2,3,4]])
print(b.ndim, ':: Dimension')
print(b.size,':: size')
print(b.shape, ':: Shape')
print(b.nbytes, ':: bytes')
print(b)

# %%
c = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
print(c[0,:])
print(c[:,2])
print(c[1,1])
# %%
# a[startIndex:endIndex: stepsize]
print(c[0,1:5:2])

# %%
nums = np.arange(100)
print(nums)
# %%
crr = np.array([[1,2,3,4,5,6], [7,8,9,10,11,12]])
crr = crr.reshape(1,-1)
print(crr)
crr = crr.reshape(4,3)
print(crr)
crr = np.transpose(crr)
print(crr)


# %%
zer0 = np.zeros((3,4))
print(zer0)
onew = np.ones((3,4))
print(onew)
fuk = np.full((3,4), 2)
print(fuk)
# %%
np.random.rand(4,2)
# %%
np.random.randint(-4,8, size=(3,3))
# %%
## Broadcasting and vector operations of Numpy

# %%
a = np.arange(4)
# %%
print(a)
# %%
a= a+10
# %%
b = np.array([10,10,10,10])
print(a*b)
# %%
from pprint import pprint
## Shallow Copy
a = np.ones((3,3))
pprint(a)
b = a 
## Now, if i change the value of b, then the value of a will also get affected
b[1,1] = 1000

print("Array B : ")
print(b)

print("Array A : ")
print(a)
# %%
a = np.ones((3,3))
pprint(a)
b = a.copy() ## deep copy
## Now, if i change the value of b, then the value of a will also get affected
b[1,1] = 1000

print("Array B : ")
print(b)

print("Array A : ")
print(a)
# %%
import numpy as np
A = np.random.randint(1,10,(5,5))

print(A)
## Addition 
A = A + 5

print(A)
# Every Element Subraction
B = A - 5

# every element Multiplication
C = A * 5 

# Every element Division
D = A/2


## Every Element exponentiation
E = A**3

## Every element trignometric Operations
F = np.cos(A)

# %%
A = np.random.rand(3,4)
B = np.random.rand(4,3)

## Use function
C = np.matmul(A,B)

## use the symbol
C = A@B
print(C)
# %%
import matplotlib.pyplot as plt

# Create a list of data points
a = [10, 12, 13, 24, 35]

# Plot the data points
plt.plot(a)

# Add labels and title to the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Plot')

# Display the plot
plt.show()
# %%
