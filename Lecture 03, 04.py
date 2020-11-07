#!/usr/bin/env python
# coding: utf-8

# Lecture02.

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

plt.plot(x_data, y_data)
plt.xlabel('Hours')
plt.ylabel('Points')
plt.show()


# In[ ]:


w = 1.0

# x를 입력으로 받는 우리의 모델
def forward(x): # y_pred = x*w
  return x*w

def loss(x,y):
  y_pred = forward(x)
  return (y_pred-y)**2


# In[ ]:


for w in np.arange(0.0, 4.1, 0.1):
  print('w = ', w)
  l_sum = 0
  for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    l = loss(x_val, y_val)
    l_sum += l
    print('\t', x_val, y_val, y_pred_val, l)

  print('MSE = ', l_sum/3) # MSE니까 데이터가 3개 일때 3으로 나눔


# In[ ]:


w_list = []
mse_list = []

for w in np.arange(0.0,4.1,0.1):
  l_sum = 0
  for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    l = loss(x_val, y_val)
    l_sum += l
  w_list.append(w)
  mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()


# Exercise 2-1

# In[6]:


import torch
import numpy as np
import matplotlib.pyplot as plt

# 임의의 데이터
x_data = [0, 0.204081633, 0.408163265, 0.612244898, 0.816326531, 1.020408163,
          1.224489796, 1.428571429, 1.632653061, 1.836734694]
y_data = [2.064980716, 2.180084184, 3.11649903, 3.251961744, 3.313814195,
          3.773538537, 4.328553494, 4.494099566, 5.282628422, 5.732555286]

plt.scatter(x_data, y_data)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[16]:


w = 1.0

# x를 입력으로 받는 우리의 모델
def forward(x): # y_pred = x*w
  return x*w

def loss(x,y):
  y_pred = forward(x)
  return (y_pred-y)**2

w_list = []
mse_list = []

for w in np.arange(0.0,10.1,0.1):
  l_sum = 0
  for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    l = loss(x_val, y_val)
    l_sum += l
  w_list.append(w)
  mse_list.append(l_sum/len(x_data))

# 가중치 w에 따른 MSE(Loss) 값 그래프
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

print('MSE 값이 최소로 되는 지점의 가중치 w: ', w_list[np.argmin(mse_list)])


# Lecture03.

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x): # y_pred = x*w
  return x*w

def loss(x,y):
  y_pred = forward(x)
  return (y_pred-y)**2

# loss(x,y)를 w에 대해 미분한 것
def gradient(x,y):
  return 2*x*(x*w-y)

print('predict (before training)', 4, forward(4))

for epoch in range(100):
  for x_val, y_val in zip(x_data, y_data):
    grad = gradient(x_val, y_val)
    w = w - 0.01*grad
    print('\tgrad: ', x_val, y_val, grad)
    l = loss(x_val, y_val)

  print('progress: ', epoch, 'w = ', w, 'loss = ', l)

print('predict (after training)', '4 hours', forward(4))


# Exercise 3-1.

# In[18]:


import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w1 = 1.0
w2 = 1.0
b = 1.0

def forward(x): # y_pred = x*w
  return (x**2)*w2 + x*w1 + b

def loss(x,y):
  y_pred = forward(x)
  return (y_pred-y)**2

# loss(x,y)를 w에 대해 미분한 것
def gradient1(x,y):
  return 2*x*(w2*(x**2)+x*w1-y+b)

def gradient2(x,y):
  return 2*(x**2)*(w2*(x**2)+w1*x-y+b)

print('predict (before training)', 4, forward(4))

for epoch in range(100):
  for x_val, y_val in zip(x_data, y_data):
    grad1 = gradient1(x_val, y_val)
    grad2 = gradient2(x_val, y_val)
    w1 = w1 - 0.01*grad1
    w2 = w2 - 0.01*grad2
    print('\tgrad1: ', x_val, y_val, grad1)
    print('\tgrad2: ', x_val, y_val, grad2)
    l = loss(x_val, y_val)

  print('\nprogress: ', epoch, 'w1 = ', w1, 'w2 = ', w2, 'loss = ', l)

print('\npredict (after training)', '4 hours', forward(4))

