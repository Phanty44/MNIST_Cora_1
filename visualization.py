from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
from skimage.measure import block_reduce


def visualize_hog(array):
    image = array[1].reshape((28, 28, 1))

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                        channel_axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def visualize_maxpool(array):
    image = array[1].reshape((28, 28, 1))

    maxpool_image = block_reduce(image, block_size=(1, 2, 2), func=np.max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=False)
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray, aspect='auto')
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(maxpool_image, cmap=plt.cm.gray)
    ax2.set_title('MaxPooling')
    plt.show()


def visualize_wrong(predictions, y_test, array):
    misclassified = np.where(predictions != y_test)[0]
    print("Misclassified samples:")
    for i in misclassified:
        # print("Index:", i, "True label:", y_test[i], "Predicted label:", predictions[i])
        image = array[i].reshape((28, 28, 1))

        plt.imshow(image, cmap=plt.cm.gray, aspect='auto')
        plt.title('Index:' + str(i) + ' Label:' + str(y_test[i]) + ' Predicted:' + str(predictions[i]))
        plt.show()
