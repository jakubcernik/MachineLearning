"""Download the Roboflow banana ripeness dataset."""

import os
from pathlib import Path

from roboflow import Roboflow


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY before downloading the dataset")
    dataset = (
        Roboflow(api_key=api_key)
        .workspace("roboflow-universe-projects")
        .project("banana-ripeness-classification")
        .version(4)
        .download("folder", location=str(Path("roboflow")))
    )
    print(f"Dataset downloaded to {dataset.location}")


if __name__ == "__main__":
    main()
