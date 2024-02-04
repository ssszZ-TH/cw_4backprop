import os
import numpy as np
import math

os.system('cls')

def sigmoid(x):
    return (math.e**x)/(1+math.e**x)

#เราต้องการหาว่า weight ตอนจบที่จจะทำให้ได้ error น้อยที่สุด คือ weight เเต่ละอันเท่าไร

# row is feature
## column is data
data = np.array([
    [1,0,1,0]
], dtype=np.float64)

# นัำหนักความสำคัญของเเต่ละ feature
weight= np.array([
    1,
    1,
    1,
    1
])

#ผลเฉลย
y = np.array([[
    1,0,1,1
]])

#res = np.matmul(data, weight)
#print (res)

n = len(data)
b = 0.1
y = weight
print(data)#shape of data
print(weight)#shape of weight
del_w1 = (2/n)*np.matmul(np.matmul(data,weight) + b - y, data.T[0])
print(del_w1)