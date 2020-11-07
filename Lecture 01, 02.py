#!/usr/bin/env python
# coding: utf-8

# Lecture04.

# In[13]:


# Auto Gradient

import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.tensor([1.0], requires_grad=True)

# our model forward pass
def forward(x,b):
  return x * w

# loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# before training
print('predict (before training)', 4, forward(4,b).item())

for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val,b) # 1. Forward pass
    l = loss(y_pred, y_val) # 2. Compute loss
    l.backward() # 3. Back Propagation to update weights - 이 메소드가 loss의 미분값을 다 계산
    print('\tgrad: ', x_val, y_val, w.grad.item()) # - w에 대한 loss의 미분값이 w.grad.item()에 저장됨
    w.data = w.data - 0.01 * w.grad.item()

    # Manually zero the gradients after updating weights
    w.grad.data.zero_()

  print(f'Epoch: {epoch} | Loss: {l.item()}')

# After training
print('Prediction (after training)', '4hours', forward(4,b).item())


# In[9]:


# Manual Gradient

# Training Data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0  # a random guess: random value

# our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)

# before training
print('predict (before training)', 4, forward(4))

# Training loop
for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    grad = gradient(x_val, y_val)
    w = w - 0.01*grad
    print('\tgrad: ', x_val, y_val, grad)
    l = loss(x_val, y_val)
  print('progress: ', epoch, l)

# After training
print('Prediction (after training)', '4hours', forward(4))


# Exercise 4-1, 4-2

# In[15]:


import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([2.0], requires_grad=True)

# our model forward pass
def forward(x):
  return x * w + b

# loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# before training
print('predict (before training)', 4, forward(4).item())

for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val) # 1. Forward pass
    l = loss(y_pred, y_val) # 2. Compute loss
    l.backward() # 3. Back Propagation to update weights - 이 메소드가 loss의 미분값을 다 계산
    print('\tgrad: ', x_val, y_val, w.grad.item()) # - w에 대한 loss의 미분값이 w.grad.item()에 저장됨
    w.data = w.data - 0.01 * w.grad.item()

    # Manually zero the gradients after updating weights
    w.grad.data.zero_()

  print(f'Epoch: {epoch} | Loss: {l.item()}')

# After training
print('Prediction (after training)', '4hours', forward(4).item())


# Exercise 4-3

# In[28]:


import numpy as np

x = np.array([2.0])
w = np.array([1.0])
y = np.array([4.0])

y_hat = np.dot(x,w)
s = y_hat-y
loss = s**2

grad_for_s = 2*s
grad_for_y_hat = grad_for_s * 1
grad_for_w = grad_for_y_hat * x

print(grad_for_w)


# Exercise 4-5

# In[23]:


import torch
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w1 = torch.tensor([1.0], requires_grad=True) # w1 = 1이라고 가정
w2 = torch.tensor([1.0], requires_grad=True) # w2 = 1이라고 가정
b = torch.tensor([2.0], requires_grad=True) # b = 2라고 가정

# our model forward pass
def forward(x):
  return (x**2) * w2 + x * w1 + b

# loss function
def loss(y_pred, y_val):
    return (y_pred - y_val) ** 2

# before training
print('predict (before training)', 4, forward(4).item())

for epoch in range(10):
  for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val) # 1. Forward pass
    l = loss(y_pred, y_val) # 2. Compute loss
    l.backward() # 3. Back Propagation to update weights - 이 메소드가 loss의 미분값을 다 계산
    print('\tgrad1 & grad2: ', x_val, y_val, w1.grad.item(), w2.grad.item())
    w1.data = w1.data - 0.01 * w1.grad.item()
    w2.data = w2.data - 0.01 * w2.grad.item()

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()

  print(f'Epoch: {epoch} | Loss: {l.item()}')

# After training
print('Prediction (after training)', '4hours', forward(4).item())


# In[ ]:




