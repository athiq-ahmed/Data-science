from __future__ import print_function
import torch

# Uninitialized matrix
x = torch.empty(5,3); x

# Randomly initialized matrix
x = torch.rand(5,3); x

# Matrix filled zeros and dtype long
x = torch.zeros(5, 3, dtype=torch.long); x

# Construct a tensor directly from data
x = torch.tensor([5.5,3]); x

# Create a tensor based on the existing tensor
x = x.new_ones(5 ,3, dtype = torch.double); x # new_* methods take in sizes

x = torch.randn_like(x, dtype = torch.float); x # override dtype! , # result has the same size

x.size()

# Operations
# Addition
y = torch.rand(5,3)
print(x+y)  #Method 1

print(torch.add(x,y)) #Method 2

result = torch.empty(5,3)
torch.add(x, y, out=result)
print(result)  #Method 3; providing an output tensor as argument

y.add_(x)

print(x[0:1])

# resize/reshape tensor using view
x = torch.randn(4, 4);x
y = x.view(16); y
z = x.view(-1, 8); z  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

x = torch.randn(1); x
x.item()

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5); a
b = a.numpy(); b

a.add_(1)
print(a)
print(b)

# Converting NumPy Array to Torch Tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CUDA Tensors
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

