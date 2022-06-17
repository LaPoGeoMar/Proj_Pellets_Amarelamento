from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

fnames = glob("01-masks/*.npy")

for fname in fnames:
    pellet = Path(fname).stem
    img = io.imread(f"00-images/{pellet}.jpg")
    maskr = np.load(f"01-masks/{pellet}.npy")

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    fig.suptitle(pellet)

    ax0.imshow(img)

    masked = img.copy()
    masked[~maskr] = 255
    ax1.imshow(masked)

    saved = f"temp/check-{pellet}.png"
    print(f"Saving check image {saved}.")
    fig.savefig(saved)
    plt.close()
