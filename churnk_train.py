from pathlib import Path
import shutil
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


DATASET_ROOT = "/home/semawit/cylinder_internal_defect"
CHUNK_SIZE = 500


def create_chunk(images, chunk_dir):
    train_good = chunk_dir / "train" / "good"
    test_good = chunk_dir / "test" / "good"
    test_defect = chunk_dir / "test" / "defect"

    train_good.mkdir(parents=True, exist_ok=True)
    test_good.mkdir(parents=True, exist_ok=True)
    test_defect.mkdir(parents=True, exist_ok=True)

    # Copy training images
    for image in images:
        shutil.copy2(image, train_good / image.name)

    # Copy test folders
    original_test_good = (
        Path(DATASET_ROOT) / "test" / "good"
    )
    original_test_defect = (
        Path(DATASET_ROOT) / "test" / "defect"
    )

    for f in original_test_good.iterdir():
        if f.is_file():
            shutil.copy2(f, test_good / f.name)

    for f in original_test_defect.iterdir():
        if f.is_file():
            shutil.copy2(f, test_defect / f.name)


def train_chunk(chunk_path, chunk_id):

    datamodule = Folder(
        name=f"chunk_{chunk_id}",
        root=str(chunk_path),
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=2,
    )

    model = Patchcore()

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    engine = Engine(
        accelerator=accelerator,
        devices=1,
    )

    print(f"\nTraining chunk {chunk_id}")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print(f"Finished chunk {chunk_id}")


def main():

    train_dir = (
        Path(DATASET_ROOT)
        / "train"
        / "good"
    )

    images = sorted(
        [
            f for f in train_dir.iterdir()
            if f.is_file()
        ]
    )

    total = len(images)

    print(f"Found {total} training images")

    chunk_num = 1

    for start in range(0, total, CHUNK_SIZE):

        end = min(start + CHUNK_SIZE, total)

        current_images = images[start:end]

        chunk_dir = Path(
            f"/tmp/patchcore_chunk_{chunk_num}"
        )

        if chunk_dir.exists():
            shutil.rmtree(chunk_dir)

        create_chunk(
            current_images,
            chunk_dir,
        )

        train_chunk(
            chunk_dir,
            chunk_num,
        )

        shutil.rmtree(chunk_dir)

        torch.cuda.empty_cache()

        chunk_num += 1


if __name__ == "__main__":
    main()