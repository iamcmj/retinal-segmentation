import numpy as np
import matplotlib.pyplot as plt

img_path = "dataset/DRIVE/processed/training/images/21_training.npy"
mask_path = "dataset/DRIVE/processed/training/masks/21_training.npy"

img = np.load(img_path)     # shape: (256, 256, 3)
mask = np.load(mask_path)   # shape: (256, 256)

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)
print("Image max/min:", img.max(), img.min())
print("Mask unique values:", np.unique(mask))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.title("Processed Image (CLAHE + Green Channel)")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Processed Mask (Binary)")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()