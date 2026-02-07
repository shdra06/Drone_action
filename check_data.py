import numpy as np
import matplotlib.pyplot as plt
import os

files = os.listdir("dataset")
sample = np.load("dataset/" + files[0])

print("Sample shape:", sample.shape)

plt.figure(figsize=(6,2))
for i in range(sample.shape[-1]):
    plt.subplot(1, sample.shape[-1], i+1)
    plt.imshow(sample[:,:,i], cmap="gray")
    plt.axis("off")
plt.show()
