from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io


def get_dominant_color(pil_img, palette_size=16):
    # Resize image to speed up processing
    img = pil_img.copy()
    img.thumbnail((100, 100))

    # Reduce colors (uses k-means internally)
    paletted = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=palette_size)

    # Find the color that occurs most often
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    idx = 1  # We want the second one to avoid getting the mask, otherwise use 0.
    palette_index = color_counts[idx][
        1
    ]  # We want the second one to avoid getting the mask
    dominant_color = palette[palette_index * 3 : palette_index * 3 + 3]

    return dominant_color


fnames = glob("01-masks/*.npy")


dataset = {}
for fname in fnames:
    pellet = Path(fname).stem
    img = io.imread(f"00-images/{pellet}.jpg")
    maskr = np.load(f"01-masks/{pellet}.npy")

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(15, 5))
    fig.suptitle(pellet)

    ax0.imshow(img)

    # Remove background
    masked = img.copy()
    masked[~maskr] = 255
    ax1.imshow(masked)

    # Dominant color
    pil_img = Image.fromarray(masked)

    R, G, B = get_dominant_color(pil_img)
    dataset.update({pellet: (R, G, B)})

    colored = masked.copy()
    colored[maskr, 0] = R
    colored[maskr, 1] = G
    colored[maskr, 2] = B
    ax2.imshow(colored)

    saved = f"temp/check-{pellet}.png"
    print(f"Saving check image {saved}.")
    fig.savefig(saved)
    plt.close()

# Save final data
df = pd.DataFrame(dataset, index=("R", "G", "B")).T
df.to_csv("RGB_values.csv")
