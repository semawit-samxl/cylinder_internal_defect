from pathlib import Path

import torch

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.loggers import AnomalibTensorBoardLogger

from lightning.pytorch.callbacks import ModelCheckpoint


def main():

  
    for folder in ["logs", "checkpoints", "results", "models"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

   
    # Device Information
   
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"Accelerator : {accelerator}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"CUDA        : {torch.version.cuda}")
        print(
            f"VRAM        : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 60)

    
    # Dataset
  
    datamodule = Folder(
        name="cylinder_internal_defect",
        root="./cylinder_internal_defect",
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",

        # Increase gradually if GPU memory allows
        train_batch_size=128,
        eval_batch_size=128,

        num_workers=8,
    )

  
    # Model
   
    model = Patchcore()

   
    # TensorBoard Logger
    
    logger = AnomalibTensorBoardLogger(
        save_dir="logs",
        name="patchcore_cylinder_internal_defect",
    )

   
    # Checkpoint Saving
   
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="patchcore-{epoch:02d}",
        save_last=True,
        save_top_k=1,
        monitor=None,
    )

   
    # Engine
  
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

   
    # Training
   
    print("\nStarting PatchCore training...\n")

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nTraining completed successfully!")

  
    # Testing
   
    print("\nStarting evaluation...\n")

    results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nEvaluation Results:")
    print(results)

    print("\nSaved Artifacts:")
    print("logs/")
    print("checkpoints/")
    print("results/")
    print("models/")


if __name__ == "__main__":
    main()