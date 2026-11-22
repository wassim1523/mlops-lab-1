import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image

if __package__:
    from .data import CLASSES, SPLITS
else:
    from data import CLASSES, SPLITS


def validate(root, minimum=50, tau=0.05):
    errors = {
        check: [] for check in ("folders", "images", "minimum", "leakage", "balance")
    }
    counts = {split: Counter() for split in SPLITS}
    seen = {}
    for split in SPLITS:
        folder = root / split
        folders = (
            {p.name for p in folder.iterdir() if p.is_dir()}
            if folder.is_dir()
            else set()
        )
        if folders != set(CLASSES):
            errors["folders"].append(f"{split}: {sorted(folders ^ set(CLASSES))}")
        for label in CLASSES:
            for path in sorted((folder / label).glob("*")):
                try:
                    digest = hashlib.md5(path.read_bytes()).digest()
                    if seen.setdefault(digest, split) != split:
                        errors["leakage"].append(str(path))
                    with Image.open(path) as img:
                        img.load()
                        if img.mode != "RGB" or img.size != (128, 128):
                            raise ValueError("expected RGB 128x128")
                    counts[split][label] += 1
                except (ValueError, OSError) as exc:
                    errors["images"].append(f"{path}: {exc}")
            if counts[split][label] < minimum:
                errors["minimum"].append(f"{split}/{label}: {counts[split][label]}")
    train, val = counts["training"], counts["validation"]
    for label in CLASSES:
        if (
            not train.total()
            or not val.total()
            or abs(train[label] / train.total() - val[label] / val.total()) > tau
        ):
            errors["balance"].append(label)
    for check, issues in errors.items():
        print(
            f"{check}: " + (f"FAIL ({len(issues)}): {issues[0]}" if issues else "PASS")
        )
    return not any(errors.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--tau", type=float, default=0.05)
    args = parser.parse_args()
    if args.min_per_class < 1 or not 0 <= args.tau <= 1:
        parser.error("minimum must be positive and tau must be in [0, 1]")
    raise SystemExit(0 if validate(args.dataset, args.min_per_class, args.tau) else 1)
