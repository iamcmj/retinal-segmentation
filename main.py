def main():
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    
    path = "data/Synapse/test_vol_h5/case0001.npy.h5"

    with h5py.File(path, "r") as f:
        print("Keys:", list(f.keys()))
        img = np.array(f["image"])
        label = np.array(f["label"])

    print("Image shape:", img.shape)
    print("Label shape:", label.shape)
    
    slice_idx = 70  

    plt.subplot(1,2,1)
    plt.imshow(img[slice_idx], cmap='gray')
    plt.title("CT slice")

    plt.subplot(1,2,2)
    plt.imshow(label[slice_idx])
    plt.title("Segmentation mask")
    plt.show()

if __name__ == "__main__":
    main()
