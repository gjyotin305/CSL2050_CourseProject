from sklearn.decomposition import PCA
import numpy as np


def pca_for_rgb(img, n_components):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    try:
        pca = PCA(n_components=n_components)

        red_transformed = pca.fit_transform(red)
        red_inverted = pca.inverse_transform(red_transformed)

        green_transformed = pca.fit_transform(green)
        green_inverted = pca.inverse_transform(green_transformed)

        blue_transformed = pca.fit_transform(blue)
        blue_inverted = pca.inverse_transform(blue_transformed)
        img_compressed = (
            np.dstack(
                (red_inverted,
                 green_inverted,
                 blue_inverted))).astype(
            np.uint8)
        return img_compressed
    except BaseException:
        print("Kindly put valid number of components")
