from re import T
import numpy as np
zeros = np.zeros(5)

print(zeros)
print(zeros.shape)

nums = np.arange(100)
print(nums)

nums = np.arange(100)
print(nums.shape)
nums = nums.reshape( (10,10) )
print(nums)



a = np.array([1,2,3])
print(a)
print(a.shape)
print(a.dtype)
print(a.ndim)
print(a.size)
print(a[0])
a[0] = 10
print(a[0])

t = 10
b = np.array([2, 4, 1, 3])
c = t * b
print(c)



#array vs lists
l = [1, 2, 3]
r = np.array([4, 5, 6])
print(l)
print(r)
s = l + [4]
print(s)
b = r + [4]
print(b)



l1 = [1,2,3]
l2 = [4,5,6]
a1 = np.array(l1)
a2 = np.array(l2)
b1 = np.dot(a1, a2)
print(b1)
c1 = a1 @ a2
print(c1)

#multidimentional array

a3 = np.array([[1, 2], [3, 4]])
print(a3.shape)
print(a3.ndim)
print(a3[0])
print(a3[0] [0]) #primo elemento della prima riga
print(a3[1] [0])

print(np.linalg.inv(a3))
print(np.linalg.det(a3))
print(np.diag(a3))

#indexing and slicing
k = np.array([[1, 2, 3, 5], [6, 7, 8, 9]])
print(k)
print(k[:, 0]) # : considera tutti gli elementi delle righe
print(k[1, :])
print(k[-1, -2])
bool_index = k > 2
print(bool_index)
k = np.array([20, 31, 24, 52, 61, 72])
print(k)
b = [1, 4, 5] # riconosce il numero di posizione dei numeri
print(k[b])
c = np.argwhere(k%2==0).flatten()
print(k[c])

a = np.eye(3) #identifica una matrice in questo caso 3 --> 3 righe
print(a)

a = np.zeros(10) # moltipica il 0 per il numero tra parentesi
print(a)
a = np.ones([10, 10])
print(a)

#arange and reshape
a = np.arange(1, 7) #cera un array fino al 6
print(a)
b = a.reshape((2, 3))
print(b)

b = a[np.newaxis, :]
print(b)
b = a[:, np.newaxis]
print(b)
print(b.shape)

#concatenation
a = np.array([[1,2],[2,4]])
b = np.array([[5, 6]])
c = np.concatenate((a,b))
print(c)

c = np.concatenate((a,b), axis = None)
print(c)
a = np.array([1, 2, 3, 4])
b = np.array([4, 5, 6, 7])
c = np.hstack((a,b))
print(c)
c = np.vstack((a,b))
print(c)