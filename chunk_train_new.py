from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch

DATASET_ROOT = "/home/semawit/cylinder_internal_defect"

def main():

    datamodule = Folder(
        name="cylinder_defect",
        root=DATASET_ROOT,
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=1,
        image_size=(320, 320),  # adjust if needed
    )

    model = Patchcore(
        backbone="resnet18",
        coreset_sampling_ratio=0.001,
    )

    engine = Engine(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    print("Training...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("Testing...")
    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nResults:")
    print(results)

if __name__ == "__main__":
    main()