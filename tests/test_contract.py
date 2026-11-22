import shutil

from PIL import Image

from src.food11.data import CLASSES, SPLITS
from src.food11.promote import gate
from src.food11.validate_data import validate


def test_data_contract(tmp_path, capsys):
    for split_id, split in enumerate(SPLITS):
        for class_id, label in enumerate(CLASSES):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            Image.new("RGB", (128, 128), (class_id * 10, split_id * 50, 0)).save(
                folder / "one.png"
            )
    assert validate(tmp_path, minimum=1)
    shutil.copy(
        tmp_path / "training/Bread/one.png", tmp_path / "evaluation/Bread/leak.png"
    )
    assert not validate(tmp_path, minimum=1)
    assert "leakage: FAIL" in capsys.readouterr().out


def test_promotion_threshold():
    assert gate(0.81, 0.8, 1000)[0] == "challenger"
    assert gate(0.83, 0.8, 1000)[0] == "champion"
