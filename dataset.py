import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CustomImageSet(Dataset):
    def __init__(self, img_dir, number_of_img=None, transform=None):
        """
        This is a class representing a custom dataset.

        Parameters
        ----------
        img_dir : str
            Directory with all the images.
        number_of_img : int, optional
            Number of images to load per category. If None, load all the images.
        transform :
            Transformations to be applied to all the images. If None, no transformation is applied.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.category_names = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}
        self.image_paths = []
        self.labels = []
        self.load_images(number_of_img)

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
        -------
        int
            Number of images.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """
        Returns an image and its label.

        Parameters
        ----------
        index: int
            The index of the image to return.

        Returns
        -------
        tuple
            A tuple containing the image and its label.
        """
        img_path = os.path.join(self.img_dir, self.image_paths[index])
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]

        return image, label

    def load_images(self, n=None):
        """
        Load images from a directory.

        Parameters
        ----------
        n : int
            Number of images to load per category. If None, load all the images.
        """
        for category_key, category in self.category_names.items():
            category_path = os.path.join(self.img_dir, category)
            image_files = os.listdir(category_path)

            if n is not None:
                image_files = image_files[:n]

            for image in image_files:
                self.image_paths.append(os.path.join(category_path, image))
                self.labels.append(category_key)


class OCT5kDataset(Dataset):
    def __init__(self, image_folder, labels_dict, transform=None):
        self.image_folder = image_folder
        self.labels_dict = labels_dict
        self.images = list(labels_dict.keys())
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        label = np.array(self.labels_dict[img_name])

        if self.transform:
            image = self.transform(image)

        return image, label


def count_images(path):
    """
    Count the number of images in each category.

    Parameter
    --------
    path: str
        Path to the directory containing all the categories.
    """
    for category in os.listdir(path):
        count = 0
        category_path = os.path.join(path, category)
        for _ in os.listdir(category_path):
            count += 1
        print(f'In {category} folder are {count} images.')
