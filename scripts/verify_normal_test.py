from pathlib import Path
import cv2

folder = Path("data/raw/bottle/test/good")
files = sorted(folder.glob("*.png"))

valid = 0
for file in files:
    image = cv2.imread(str(file))
    if image is not None:
        valid += 1
        print(f"OK   {file.name} | shape={image.shape}")
    else:
        print(f"FAIL {file.name}")

print(f"\nValid normal test images: {valid}/{len(files)}")
