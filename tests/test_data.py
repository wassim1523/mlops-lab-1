from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


def test_labels():
    assert len(CLASSES) == 11
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("3_120.jpg") == "Egg"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


def test_resize():
    img = resize_image(Image.new("L", (512, 384)))
    assert img.mode == "RGB"
    assert img.size == (128, 128)
