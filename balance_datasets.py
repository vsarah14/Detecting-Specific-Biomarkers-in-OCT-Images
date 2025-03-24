import os
import shutil
import random


def move_images(category, train_dir, val_dir, num_images):
    """
        Move a specified number of images from the training directory to the validation/test directory for a given category.

        Parameters
        ----------
        category : str
            The name of the category (e.g., 'CNV', 'DME', 'DRUSEN', 'NORMAL').
        train_dir : str
            The path to the training directory.
        val_dir : str
            The path to the validation directory.
        num_images : int
            The number of images to move from the training directory to the validation directory.
        """
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)

    if not os.path.exists(val_category_dir):
        os.makedirs(val_category_dir)

    images = os.listdir(train_category_dir)
    images_to_move = random.sample(images, min(num_images, len(images)))

    for image in images_to_move:
        src = os.path.join(train_category_dir, image)
        dst = os.path.join(val_category_dir, image)
        shutil.move(src, dst)


train_directory = "C:/Users/voicu/OneDrive/Desktop/licenta/OCT2017/train"
val_directory = "C:/Users/voicu/OneDrive/Desktop/licenta/OCT2017/val"
test_directory = "C:/Users/voicu/OneDrive/Desktop/licenta/OCT2017/test"
number_of_images_to_move = 1

category = "DME"
move_images(category, train_directory, val_directory, 100)
