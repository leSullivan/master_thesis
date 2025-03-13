import os
import random
from shutil import copy2
from PIL import Image
import torchvision.transforms as transforms

from src.config import FENCE_IMG_PATH, BG_IMG_PATH, TRAIN_IMG_PATH, SEED

random.seed(SEED)


class RandomZoom:
    def __init__(self, scale=(0.9, 1.0)):
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        crop_scale = random.uniform(*self.scale)

        crop_width = int(width * crop_scale)
        crop_height = int(height * crop_scale)

        # Randomly select a crop
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Resize the cropped image back to original dimensions
        resized_image = cropped_image.resize((width, height), Image.LANCZOS)

        return resized_image


# Define the augmentation pipeline
augmentation_pipeline = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        RandomZoom(scale=(0.95, 1.0)),
    ]
)


def augment_image(image):
    augmented_image = augmentation_pipeline(image)
    return augmented_image


def sample_images():
    train_A_path = os.path.join(TRAIN_IMG_PATH, "trainA")
    train_B_path = os.path.join(TRAIN_IMG_PATH, "trainB")
    test_A_path = os.path.join(TRAIN_IMG_PATH, "testA")

    if not os.path.exists(train_A_path):
        os.makedirs(train_A_path)

    if not os.path.exists(train_B_path):
        os.makedirs(train_B_path)

    if not os.path.exists(test_A_path):
        os.makedirs(test_A_path)

    # Clear the directories of previous images
    for path in [train_A_path, train_B_path, test_A_path]:
        filelist = [f for f in os.listdir(path) if f.endswith(".jpg")]
        for f in filelist:
            os.remove(os.path.join(path, f))

    # Get lists of available images
    background_images = [
        os.path.join(BG_IMG_PATH, f)
        for f in os.listdir(BG_IMG_PATH)
        if f.endswith("jpg")
    ]

    fence_images = []
    for subdir in os.listdir(FENCE_IMG_PATH):
        subdir_path = os.path.join(FENCE_IMG_PATH, subdir)
        if os.path.isdir(subdir_path):
            fence_images.extend(
                [
                    os.path.join(subdir_path, f)
                    for f in os.listdir(subdir_path)
                    if f.endswith("jpg")
                ]
            )

    # Determine the number of samples based on the fence images
    num_samples = len(fence_images)

    # Pick two random background images for the test set
    test_background_images = random.sample(background_images, 2)

    for i, image_path in enumerate(test_background_images):
        _, filename = os.path.split(image_path)
        dest_path = os.path.join(test_A_path, f"{i}_{filename}")
        copy2(image_path, dest_path)

    # Remove the test images from the background image list
    remaining_background_images = list(
        set(background_images) - set(test_background_images)
    )

    # If there are fewer remaining background images than fence images, sample with replacement
    sampled_background = random.choices(remaining_background_images, k=num_samples)

    # Augment and save background images with labeling
    for i, image_path in enumerate(sampled_background):
        _, filename = os.path.split(image_path)
        dest_path = os.path.join(train_A_path, f"{i}_augmented_{filename}")

        # Load and augment the background image
        with Image.open(image_path) as img:
            augmented_img = augment_image(img)
            augmented_img.save(dest_path)

    # Copy fence images to train_B without augmentation
    for i, image_path in enumerate(fence_images):
        _, filename = os.path.split(image_path)
        dest_path = os.path.join(train_B_path, f"{i}_{filename}")
        copy2(image_path, dest_path)

    print(
        f"Successfully augmented and saved {num_samples} background images in train_A, copied {num_samples} fence images to train_B, and moved 2 images to test_A."
    )


if __name__ == "__main__":
    sample_images()
