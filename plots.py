import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from umap import UMAP

category_names = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}


def display_n_images(n, dataset):
    """
    Display n images from the dataset.

    Parameters
    ----------
    n : int
        Number of images to display.
    dataset : Dataset
        The dataset from which to load the images.
    """
    plt.figure(figsize=(15, 6))
    for i in range(n):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()

        # check the data type and clip values if they are floats
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image, 0, 1)

        if n % 2 == 0:
            plt.subplot(2, int(n / 2), i + 1)
        else:
            plt.subplot(2, int(n / 2 + 1), i + 1)
        plt.imshow(image)
        plt.title(f'Label: {label}')

    plt.show()


def plot_confusion_matrix(labels, predictions):
    """
    Plot a confusion matrix.

    Parameters
    ----------
    labels : np.ndarray
        True labels of the images.
    predictions : np.ndarray
        Predicted labels of the images.
    """
    conf_matrix = confusion_matrix(labels, predictions)
    class_names = [category_names[i] for i in range(len(category_names))]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.title('Confusion Matrix')
    plt.show()


def plot_features(features, labels, true_labels, category_names):
    """
    Plot extracted features using UMAP with true labels.

    features: np.ndarray
        Features of the images.
    labels: np.ndarray
        Cluster labels corresponding to the features.
    true_labels: np.ndarray
        True labels corresponding to the features.
    category_names: dict
        Dictionary mapping label indices to category names.
    """
    umap = UMAP(n_components=2)
    feature_coord = umap.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feature_coord[:, 0], feature_coord[:, 1], c=true_labels, cmap='viridis', s=10)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
               for i in category_names.keys()]
    plt.legend(handles, category_names.values(), title="Categories")

    plt.title('UMAP Projection of Features with True Labels')
    plt.colorbar(scatter)
    plt.tight_layout()
    plt.show()


def plot_biomarkers(features, cluster_labels):
    """
    Plot extracted features using UMAP with cluster labels.

    features: np.ndarray
        Features of the images.
    cluster_labels: np.ndarray
        Cluster labels from the clustering algorithm.
    """
    umap = UMAP(n_components=2)
    feature_coord = umap.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(feature_coord[:, 0], feature_coord[:, 1], c=cluster_labels, cmap='viridis', s=10)

    unique_labels = np.unique(cluster_labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(label / max(unique_labels)),
                          markersize=10, label=f'Cluster {label}') for label in unique_labels]

    plt.legend(title='Categories', handles=handles, loc='best', frameon=True)

    plt.title('UMAP Projection of Features with Cluster Labels')
    plt.tight_layout()
    plt.show()
