import matplotlib.pyplot as plt
from skimage import data
from sklearn.decomposition import PCA
import numpy as np


image = data.camera()


image_reshaped = image.reshape(image.shape[0], -1)


def apply_pca(image, n_components):

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(transformed)
    return reconstructed


components = [5, 20, 50, 100] 
reconstructed_images = [apply_pca(image_reshaped, n) for n in components]


fig, axes = plt.subplots(1, len(components) + 1, figsize=(15, 5))


axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

for i, (comp, rec_img) in enumerate(zip(components, reconstructed_images)):
    axes[i + 1].imshow(rec_img.reshape(image.shape), cmap='gray')
    axes[i + 1].set_title(f'{comp} Components')
    axes[i + 1].axis('off')

plt.show()
