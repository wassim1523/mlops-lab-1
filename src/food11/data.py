"""Prepare the Food-11 dataset for image classification.

Run from the repository root with:
    uv run python ./src/food11/data.py
"""

from pathlib import Path
import re
import shutil

from PIL import Image, ImageOps, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "food11_raw"
PROCESSED_ROOT = DATA_ROOT / "food11_processed"
MINI_ROOT = DATA_ROOT / "food11_processed_mini"

SPLITS = ("training", "evaluation", "validation")
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}


def category_from_filename(filename: str) -> str:
    """Return the Food-11 category encoded at the start of a filename."""
    match = re.match(r"^(\d+)", filename)
    if match is None:
        raise ValueError("filename does not begin with a category number")

    category_number = int(match.group(1))
    if category_number not in CATEGORIES:
        raise ValueError(f"unknown category number {category_number}")
    return CATEGORIES[category_number]


def create_output_directories() -> None:
    """Create an empty, predictable category structure for both outputs."""
    for output_root in (PROCESSED_ROOT, MINI_ROOT):
        if output_root.exists():
            shutil.rmtree(output_root)
        for split in SPLITS:
            for category in CATEGORIES.values():
                (output_root / split / category).mkdir(parents=True, exist_ok=True)


def process_image(source: Path, destination: Path) -> None:
    """Correct orientation, convert to RGB, resize and save one image."""
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        image.save(destination, format="JPEG", quality=90, optimize=True)


def prepare_split(split: str) -> tuple[int, int, int]:
    """Process one split and return processed, mini and skipped counts."""
    source_directory = RAW_ROOT / split
    if not source_directory.is_dir():
        raise FileNotFoundError(f"Missing input directory: {source_directory}")

    mini_counts = {category: 0 for category in CATEGORIES.values()}
    processed_count = 0
    mini_count = 0
    skipped_count = 0

    files = sorted(path for path in source_directory.iterdir() if path.is_file())
    for source in files:
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            category = category_from_filename(source.name)
            output_name = f"{source.stem}.jpg"
            processed_file = PROCESSED_ROOT / split / category / output_name
            process_image(source, processed_file)
            processed_count += 1

            if mini_counts[category] < MINI_LIMIT:
                mini_file = MINI_ROOT / split / category / output_name
                shutil.copy2(processed_file, mini_file)
                mini_counts[category] += 1
                mini_count += 1
        except (ValueError, OSError, UnidentifiedImageError) as error:
            skipped_count += 1
            print(f"Warning: skipped {source}: {error}")

    return processed_count, mini_count, skipped_count


def main() -> None:
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(
            f"Food-11 was not found at {RAW_ROOT}. "
            "Expected training, evaluation and validation directories."
        )

    create_output_directories()

    total_processed = 0
    total_mini = 0
    total_skipped = 0
    for split in SPLITS:
        processed, mini, skipped = prepare_split(split)
        total_processed += processed
        total_mini += mini
        total_skipped += skipped
        print(
            f"{split}: {processed} processed, "
            f"{mini} added to mini, {skipped} skipped"
        )

    print(
        f"Done: {total_processed} processed, "
        f"{total_mini} in mini, {total_skipped} skipped"
    )


if __name__ == "__main__":
    main()
