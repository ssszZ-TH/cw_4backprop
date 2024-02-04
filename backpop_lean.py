#โค้ดที่ให้มีไว้เพื่อฝึกการใช้ Logistic Regression 
# ในการแก้ปัญหาของการทำนายข้อมูล binary (0 หรือ 1) 
# ด้วยการใช้ฟังก์ชัน Sigmoid ในการคำนวณความน่าจะเป็น (probability) 
# ของผลลัพธ์ที่เป็น positive class (1) หรือ negative class (0) 
# ตามลำดับ
import time
import numpy as np
import math

start = time.time()

# function ที่เหมาะกับ logistic regression 
# คล้ายการตัดสินใจของมนุษย์
def sigmoid(x):
    return (math.e**x)/(1+math.e**x)

#  เป็นตัวแปรที่เก็บข้อมูล input ซึ่งเป็น matrix ขนาด 10x7
feature = np.array([  [1,1,1,0,1,1,0],
                [1,0,1,1,0,0,1],
                [1,0,0,0,1,1,0],
                [1,1,0,1,0,0,1],
                [1,1,1,0,1,0,1],
                [1,0,0,1,0,1,0],
                [1,0,1,0,1,0,1],
                [1,1,0,1,0,0,0],
                [1,0,1,0,1,1,1],
                [1,1,0,0,1,1,0]], dtype=np.float64)

# เป็นตัวแปรที่เก็บข้อมูล output (คำตอบ) ซึ่งเป็น vector ขนาด 10
classify = np.array([0,0,1,0,1,0,1,0,0,1], dtype=np.float64)
weight=np.zeros([7]) # เก็บนำหนักของเเต่ละ input node
del_w = np.ones([7]) # ค่าที่น้ำหนักจะต้องเปลี่ยน ในแต่ละรอบการฝึก 
learning_rate = 0.01
feature_len=len(feature)

i=0
max_iteration = 1000000 # คอมไม่เเรงก็ปรับเป็น 1000 พอ
error = 1000

# ทำ gradient descent โดยใช้ฟังก์ชัน Sigmoid
while (error > 0.5 and i < max_iteration):
    
    # คาดเดาเหตุการ โดยใช้ Sigmoid
    # โดยใช้ฟังก์ชัน Sigmoid กับผลคูณ dot product ของ feature และ weights.
    #  array ของ ผลการคาดเดา
    s_y_hat = sigmoid(np.matmul(feature,weight))
    
    #หา mean squared error ของเเต่ละตัวใน array ใน array คาดเดา เทียบกับคำตอบจริง
    # โดยที่ y hat คือค่าที่ ai ทำนาย
    # classify คือคำตอบจริงๆ
    # ตรงบรรทัศนี้ ไม่เข้าใจที่มาสูตร รูเเค่ว่ามันมาจาก เดา - จริง * เดา * 1-เดา ทำไปเพื่อ
    del_w = (2/feature_len) * np.matmul(  (s_y_hat - classify)*s_y_hat*(1-s_y_hat)  , feature)
    
    #ปรับค่า weights โดยลบ gradient คูณกับ learning rate.
    weight = weight - (learning_rate * del_w)
    
    #รอบการฝึก
    i += 1
    
    # ผลรวมความคาดเคลื่อน ระหว่าง ค่าจริง กับ aiเดา
    error = sum(abs(s_y_hat - classify ))


print("weight arr =",weight,"loss=",error)
feature_name = ["Open App > 20times/week",
                "Buy Toy",
                "Buy Sci-FiBook",
                "Buy TV",
                "Search for Smartphone",
                "Invite friend to used app",
                "Buy tuf gaming"]
for i,w in enumerate(weight):
    print(f"{feature_name[i]} have weight {w}")
    
# สมมุติว่ามีลูกค้าคนนึง ทำทุกอย่างเลย
new_data = [1,1,1,1,1,1,1]
forecast = sigmoid(np.matmul(new_data,weight))
print(f"this customer have chance {forecast*100}% to buy tuf gaming")

# สมมุติว่ามีลูกค้าคนนึง ทำเเบบที่ผมคิดมาเล่นๆ
new_data = [1,0,0,0,1,0,1]
forecast = sigmoid(np.matmul(new_data,weight))
print(f"this customer have chance {forecast*100}% to buy tuf gaming")


end = time.time()
print(f"\nTime taken to run the code was {end-start} seconds\n")