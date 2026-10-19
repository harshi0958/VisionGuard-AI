
from pathlib import Path
import argparse
import cv2


def inspect_image(image_path: str) -> None:
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("OpenCV could not read this image.")

    height, width, channels = image.shape

    print("\n=== VisionGuard AI: Image Inspection ===")
    print(f"File       : {path.name}")
    print(f"Resolution : {width} x {height}")
    print(f"Channels   : {channels}")
    print(f"Data type  : {image.dtype}")
    print(f"File size  : {path.stat().st_size:,} bytes")
    print(f"Mean BGR   : {image.mean(axis=(0, 1)).round(2)}")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{path.stem}_inspected.png"

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Could not save processed image.")

    print(f"Saved copy : {output_path.resolve()}")
    print("\nImage inspection completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect an image for VisionGuard AI."
    )
    parser.add_argument("image", help="Path to an image file")
    args = parser.parse_args()

    inspect_image(args.image)
