import numpy as np
import pandas as pd

####### why we use numpy? #######
# [[1,2,3],
#  [2,3,4]]
# This matrix will be built via numpy

array = np.array([[1,2,3],
                 [2,3,4]])
print(array)

# result:
#
# [[1,2,3],
#  [2,3,4]]

# check dimension
print('number of dim:', array.ndim)

# result:
#
# ('number of dim:', 2)
#
# There are 2 dimension in our array.

# check shape
print('shape:',array.shape)

# check size
print('size:',array.size)

a = np.array([2,3,4],dtype = np.float)  # int默认 int64, int32, float64, float32, 64bit is better,if you want to save space, you can use 16bit
print(a.dtype)

b = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])
print(b)

# all zero
c = np.zeros((3,4),dtype = np.int)
print(c)

# all one
d = np.ones((3,4), dtype = np.int16)
print(d)

# all empty
e = np.empty((3,4))
print(e)

# sequental list
a = np.arange(10,20,2)  # (initial, end, step)
print(a)

a = np.arange(12).reshape((3,4)) # define into matrix
print(a)

# linespace
a = np.linspace(1,10,20) # 生成一个20段的数列平均分配
print(a)

a = np.linspace(1,10,6).reshape((2,3)) # 生成一个20段的数列平均分配
print(a)

# 基础运算
a = np.array([10,20,30,40])
b = np.arange(4)

c = a-b     # minus
print(c)

c = a+b     # add
print(c)

c = b**2    # sqare
print(c)

c = 10*np.sin(a) # sin, also tan(), cos()
print(c)

print(b)
print(b<3)
print(b==3)

a = np.array([[1,1],
              [0,1]])
b = np.arange(4).reshape((2,2))

# 乘法
c = a*b  #逐个相乘
print(c)
c_dot = np.dot(a,b) #矩阵乘法
print(c_dot)
c_dot_2 = a.dot(b)  #和上面的算法一样
print(c_dot_2)

a = np.random.random((2,4))
print(a)

# MAX SUM MIN
print(np.sum(a))
print(np.min(a))
print(np.max(a))

print(a)
print(np.min(a, axis=1)) #每一行 axis=1
print(np.min(a, axis=0)) #每一列 axis=0

#运算2
A = np.arange(2,14).reshape((3,4))
print(A)

print(np.argmin(A)) #最小值索引，也就是位置

print(np.argmax(A)) #最大值索引，也就是位置

print(np.mean(A)) #平均值
print(A.mean()) #或者
print(np.average(A)) #或者
print(np.median(A)) #中位数
print(np.cumsum(A)) #每个都会和之前的相加，例如[2,3,4] 结果为[2,(2+3),(2+3+4)]
B = [2,4,7,8]
print(np.diff(A)) #每两个的差,两个数相比较，差距输出结果
print(np.diff(B))

print(np.nonzero(A)) #找出非零数，输出行和列，分别包含非零的坐标
#例如：
B = [[0,1,0,3],[1,1,1,0],[0,3,4,5]]
print(np.nonzero(B)) #输出坐标[(0,1),(0,3),(1,0),(1,1),(1,2),(2,1),(2,2),(2,3)], (行，列)

print(np.sort(B)) #排序

print(A)
print('---矩阵转向---')
print(np.transpose(A))
print(A.T)

print((A.T).dot(A)) #矩阵乘法
#print(A.dot(A))  这个是错误的因为矩阵的规格不一样

print(np.clip(A,5,9)) #所有小于5的数变成5，所有大于9的数变成9，其间不变
#选择行或者列进行计算平均值
print(np.mean(A,axis=0))

#index 索引（位置找值）
print('-------------------------index--------------------------')
A = np.arange(3,15)
print(A)
print(A[3])

A = np.arange(3,15).reshape((3,4))
print(A[2])
print(A[2][1])
print(A[2,1]) #和上面的一样

print(A[2, :]) #代表所有数
print(A[0][1:3]) #1以后到3的所有

#迭代每一行
for row in A:
    print(row)
    print('----')

print('col')
#迭代每一列
for col in A.T:
    print(col)
    print('----')

print(A.flatten())  #直接返回list
#迭代每一项
for item in A.flat: #A.flat是返回平滑器
    print(item)


#numpy array 合并
print('----------------------merge array----------------------')

A = np.array([1,1,1])
B = np.array([2,2,2])

#上下合并
print(np.vstack((A,B))) #vertical stack

C = np.vstack((A,B))
print(C.shape)

#左右合并
D = np.hstack((A,B)) # horizontal stack
print(D)
print(D.shape)

A = np.array([[1,1,1]
             ,[3,3,3]])
B = np.array([[2,2,2]
             ,[4,4,4]])

print(np.vstack((A,B)))
print(np.hstack((A,B)))  #必须要对应维度

#横向的一个数列例如[1,1,1]，不能用A.T的形式变成
#[[1]
#,[1]
#,[1]]的形式

A = np.array([1,1,1])

print(A[np.newaxis, :]) #为行加一个维度
print(A[:, np.newaxis]) #纵向加维度


A = np.array([1,1,1])[:,np.newaxis]
B = np.array([2,2,2])[:,np.newaxis]


print(np.vstack((A,B)))
print(np.hstack((A,B)))


C = np.concatenate((A,B,B,A),axis=0) #多个表合并 上下合并
D = np.concatenate((A,B,B,A),axis=1) #多个表合并 左右合并

print('-----------')
print(C)
print(D)

## 分割两个列表或者矩阵

A = np.arange(12).reshape((3,4))  #建立array
print(A)

# 横向分割 (以行分割)
print(np.split(A, 2, axis = 1))
# 注：分成两块，两个array，左半边 [[0,1],[4,5],[8,9]]，右半边[[2,3],[6,7],[10,11]]
# 相当于将4列分成1，2或4块

# 横向分割（以列分割）
print(np.split(A, 3, axis = 0))
# 注： 分成三个部分分别是第一维第二维和第三维，对应的3个array
# 相当于将3行分为1或3部分


# 如果想进行强制分割不等队列

# print(np.split(A, 3, axis=1))
print(np.array_split(A, 3, axis=1))
# 注：分割方式将4个分成3份 2，1，1分割

print(np.vsplit(A, 3))   #纵向分割
print(np.hsplit(A, 2))   #横向分割

# Numpy copy 和 deep copy

a = np.arange(4)
# array([0,1,2,3])

b = a
c = a
d = b
print(b)
print(c)
print(d)

# 注：改变a后，所有b,c,d都改变

a[0] = 11
print(a)
# 直接改变列表a中的第0项

# 判断b,c,d是否随之改变

print(b is a)
print(c is a)
print(d is a)
print(d) # 所以之前d = a导致的是与a同时变化

d[1:3] = [22,33]
print(a)
print(b)
print(c)

# 同理改变关联项，也就是=两边某个参数，关联项也会随之变化


# copy() 强制拷贝

b = a.copy() # deep copy
print(b)

a[3] = 44
print(a)
print(b)
print(b is a)

# 注：利用强制拷贝后改变a，并不能改变b，也就是说其关联项不受影响


# Pandas learning

# import pandas as pd
# import numpy as np

s = pd.Series([1,3,6,np.nan,44,1])
print(s)