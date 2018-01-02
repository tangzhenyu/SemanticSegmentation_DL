import numpy as np
from matplotlib import pyplot as plt
import cv2

def _discrete_matshow_adaptive(data, labels_names=[], title=""):

    fig_size = [7, 6]
    plt.rcParams["figure.figsize"] = fig_size
    cmap = plt.get_cmap('Paired', np.max(data)-np.min(data)+1)
    mat = plt.matshow(data,
                      cmap=cmap,
                      vmin = np.min(data)-.5,
                      vmax = np.max(data)+.5)

    cax = plt.colorbar(mat,
                       ticks=np.arange(np.min(data),np.max(data)+1))

    if labels_names:
        cax.ax.set_yticklabels(labels_names)
    
    if title:
        plt.suptitle(title, fontsize=15, fontweight='bold')

    fig = plt.gcf()
    fig.savefig('data/tmp.jpg', dpi=300)
    img = cv2.imread('data/tmp.jpg')
    return img


        
def visualize_segmentation_adaptive(predictions, segmentation_class_lut, title="Segmentation"):

    # TODO: add non-adaptive visualization function, where the colorbar
    # will be constant with names
    
    unique_classes, relabeled_image = np.unique(predictions,
                                                return_inverse=True)

    relabeled_image = relabeled_image.reshape(predictions.shape)

    labels_names = []

    for index, current_class_number in enumerate(unique_classes):

        labels_names.append(str(index) + ' ' + segmentation_class_lut[current_class_number])

    im=_discrete_matshow_adaptive(data=relabeled_image, labels_names=labels_names, title=title)
    return im