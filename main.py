"""Download dataset."""
import kagglehub


def download() -> None:
    # Download latest version
    path = kagglehub.dataset_download("shahriar26s/banana-ripeness-classification-dataset")

    print("Path to dataset files:", path)


if __name__ == "__main__":
    download()
