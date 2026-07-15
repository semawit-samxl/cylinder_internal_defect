from pathlib import Path
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger

from lightning.pytorch.callbacks import ModelCheckpoint


def main():

    # Create folders if they don't exist
    for folder in ["logs", "checkpoints", "results", "models"]:
        Path(folder).mkdir(exist_ok=True)

    # Dataset
    datamodule = Folder(
        name="cylinder_internal_defect",
        root="./cylinder_internal_defect",
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=4,
    )

    # PatchCore Model
    model = Patchcore()

    # TensorBoard Logger
    logger = AnomalibTensorBoardLogger(
        save_dir="logs",
        name="patchcore_cylinder_internal_defect",
    )

    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="patchcore-{epoch:02d}",
        save_top_k=1,
        save_last=True,
    )

    # Device Selection
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    print(f"\nUsing accelerator: {accelerator}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Engine
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Training
    print("\nStarting PatchCore training...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nTraining complete.")

    # Testing
    print("\nStarting evaluation...")
    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nResults:")
    print(results)

    print("\nArtifacts saved to:")
    print("  logs/          -> TensorBoard logs")
    print("  checkpoints/   -> Saved model checkpoints")
    print("  results/       -> Prediction outputs")
    print("  models/        -> Exported models")


if __name__ == "__main__":
    main()