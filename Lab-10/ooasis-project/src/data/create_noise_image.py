import numpy as np
from PIL import Image
from pathlib import Path

OUTPUT_PATH = Path(
    "data/inference_samples/noise.jpg"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

noise = np.random.randint(
    0,
    255,
    (224,224,3),
    dtype=np.uint8
)

image = Image.fromarray(
    noise
)

image.save(
    OUTPUT_PATH
)

print(
    f"Saved: {OUTPUT_PATH}"
)