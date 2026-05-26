import numpy as np

from app.image_augmentation import (
    augment_flip,
    augment_rotate,
    augment_brightness,
    create_augmented_images,
)


def test_augment_brightness_makes_image_brighter():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)

    brighter = augment_brightness(image, 30)

    assert brighter.mean() > image.mean()

def test_augment_brightness_makes_image_darker():
    image = np.full((100, 100, 3), 100, dtype=np.uint8)

    darker = augment_brightness(image, -30)

    assert darker.mean() < image.mean()

def test_create_augmented_images_returns_expected_keys():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    augmented = create_augmented_images(image)

    assert set(augmented.keys()) == {
        "flipped",
        "rotated_right",
        "rotated_left",
        "brighter",
        "darker",
    }