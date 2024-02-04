#โค้ดที่ให้มีไว้เพื่อฝึกการใช้ Logistic Regression 
# ในการแก้ปัญหาของการทำนายข้อมูล binary (0 หรือ 1) 
# ด้วยการใช้ฟังก์ชัน Sigmoid ในการคำนวณความน่าจะเป็น (probability) 
# ของผลลัพธ์ที่เป็น positive class (1) หรือ negative class (0) 
# ตามลำดับ

import numpy as np
import math


# function ที่เหมาะกับ logistic regression 
# คล้ายการตัดสินใจของมนุษย์
def sigmoid(x):
    return (math.e**x)/(1+math.e**x)

x = np.array([  [1,1,1,0,1,1,0],
                [1,0,1,1,0,0,1],
                [1,0,0,0,1,1,0],
                [1,1,0,1,0,0,1],
                [1,1,1,0,1,0,1],
                [1,0,0,1,0,1,0],
                [1,0,1,0,1,0,1],
                [1,1,0,1,0,0,0],
                [1,0,1,0,1,1,1],
                [1,1,0,0,1,1,0]], dtype=np.float64)

y=np.array([0,0,1,0,1,0,1,0,0,1], dtype=np.float64)
w=np.zeros([7])
del_w = np.ones([7])
alpha = 0.05
n=len(x)

i=0
max_iteration = 10000
error = 1000

while (error > 0.5 and i < max_iteration):
    s_y_hat = sigmoid(np.matmul(x,w))
    del_w = (2/n) * np.matmul((s_y_hat - y)*s_y_hat*(1-s_y_hat), x)
    w = w - (alpha * del_w)
    i += 1
    error = sum(abs(s_y_hat - y ))

print(w)

