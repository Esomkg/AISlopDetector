import tempfile
import os
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def dummy_cifake_dir():
    """Create a temporary directory with the CIFAKE structure and 4 dummy images per class."""
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("train", "test"):
            for cls in ("REAL", "FAKE"):
                d = Path(tmp) / split / cls
                d.mkdir(parents=True)
                for i in range(4):
                    img = Image.new("RGB", (32, 32), color=(i * 50, 100, 150))
                    img.save(d / f"img_{i}.jpg")
        yield tmp


def test_dataset_loads(dummy_cifake_dir):
    """CIFAKE dataset should load with correct number of samples."""
    from src.data.cifake_dataset import CIFAKE
    ds = CIFAKE(data_root=dummy_cifake_dir, split="train")
    assert len(ds) == 8  # 4 REAL + 4 FAKE


def test_dataset_returns_image_and_label(dummy_cifake_dir):
    """Each item should be (PIL.Image, int). Labels: 0=REAL, 1=FAKE."""
    from src.data.cifake_dataset import CIFAKE
    ds = CIFAKE(data_root=dummy_cifake_dir, split="train")
    img, label = ds[0]
    assert isinstance(img, Image.Image)
    assert label in (0, 1)


def test_dataset_split_train_test(dummy_cifake_dir):
    """Train and test splits should both work."""
    from src.data.cifake_dataset import CIFAKE
    train_ds = CIFAKE(data_root=dummy_cifake_dir, split="train")
    test_ds = CIFAKE(data_root=dummy_cifake_dir, split="test")
    assert len(train_ds) == 8
    assert len(test_ds) == 8


def test_transform_applies(dummy_cifake_dir):
    """Applying transforms should return a torch.Tensor of correct shape."""
    from src.data.cifake_dataset import CIFAKE
    from src.data.transforms import get_train_transforms
    import torch

    transform = get_train_transforms(image_size=224)
    ds = CIFAKE(data_root=dummy_cifake_dir, split="train", transform=transform)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)


def test_labels_balanced(dummy_cifake_dir):
    """Dataset should have equal number of REAL and FAKE samples."""
    from src.data.cifake_dataset import CIFAKE
    ds = CIFAKE(data_root=dummy_cifake_dir, split="train")
    labels = [ds[i][1] for i in range(len(ds))]
    assert labels.count(0) == labels.count(1) == 4
