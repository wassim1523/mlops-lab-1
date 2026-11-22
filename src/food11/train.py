import argparse
import os
from pathlib import Path

import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import mlflow

ROOT = Path(__file__).resolve().parents[2]


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mini", "processed"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(4)

    folder = (
        "food11_processed_mini"
        if args.dataset == "mini"
        else "food11_processed"
    )
    data_root = ROOT / "data" / folder

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_sets = {
        split: datasets.ImageFolder(data_root / split, transform=transform)
        for split in ["training", "validation", "evaluation"]
    }

    reference = image_sets["training"].class_to_idx
    if len(reference) != 11:
        raise ValueError(f"Expected 11 classes, found {len(reference)}")

    for split, dataset in image_sets.items():
        if dataset.class_to_idx != reference:
            raise ValueError(f"Different class mapping in {split}")
        print(f"{split}: {len(dataset)} images", flush=True)

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "training"),
            num_workers=0,
        )
        for split, dataset in image_sets.items()
    }

    print("Loading pretrained ResNet18...", flush=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 11)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("food11")

    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}", flush=True)

        mlflow.log_params({
            "dataset": args.dataset,
            "test_size": len(image_sets["evaluation"]),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
            "device": "cpu",
            "seed": 42,
            "image_size": 128,
        })
        mlflow.log_dict(reference, "class_to_idx.json")

        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(
                model, loaders["training"], criterion, optimizer
            )
            val_loss, val_accuracy = run_epoch(
                model, loaders["validation"], criterion
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }, step=epoch)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_accuracy={val_accuracy:.2%}",
                flush=True,
            )

        _, test_accuracy = run_epoch(
            model, loaders["evaluation"], criterion
        )
        mlflow.log_metric("test_accuracy", test_accuracy)

        print("Saving model to MLflow...", flush=True)
        model.eval()
        example = next(iter(loaders["validation"]))[0][:1].numpy()
        mlflow.pytorch.log_model(
            model,
            name="model",
            input_example=example,
        )

        print(f"Test accuracy: {test_accuracy:.2%}", flush=True)
        print(f"Finished! Run ID: {run.info.run_id}", flush=True)


if __name__ == "__main__":
    main()