from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, io, morphology, segmentation
from skimage.color import rgb2gray
from skimage.measure import label, regionprops

plt.interactive(True)


def open_image(fname):
    return io.imread(fname)


def image_segmentation(img):
    gray = rgb2gray(img)
    mask = gray > filters.threshold_otsu(gray)
    borders = segmentation.clear_border(mask).astype(np.int64)
    segmentation.mark_boundaries(gray, borders)
    label_img = label(borders)
    regions = regionprops(label_img)
    return mask, borders, regions


def find_pellet(regions):
    area = 0
    for region in regions:
        if region.area > area:
            area = region.area
            pellet = region
    return pellet


def mask_background(mask, pellet):
    threshold = pellet.area * 0.01
    return morphology.remove_small_objects(mask, pellet.area - threshold)


def show_masked(img, borders, maskr):
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 8))

    masked = img.copy()
    masked[~maskr] = 255
    ax0.imshow(masked)
    ax1.imshow(img)
    ax1.contour(borders, [0.5], colors="r")


from glob import glob

fnames = glob("00-images/*.jpg")
bad_pellets = [
    "cl1_p1_psul_deixa1_g0007",
    "cl1_p2_joaq_dunai_e0009",
    "cl1_p2_joaq_dunaiii_f0010",
    "cl1_p3_joaq_dunaiii_b0010",
    "cl1_p4_brava_deixa3_c0002",
    "cl1_p4_brava_deixa5_e0010",
]

for fname in fnames:
    pellet = Path(fname).stem
    if pellet in bad_pellets:
        print(f"Skipping pellet {pellet}.")
        continue

    img = open_image(fname)

    mask, borders, regions = image_segmentation(img)
    area = find_pellet(regions)

    maskr = mask_background(mask, area)

    print(f"Processing image {fname}.")
    savename = f"01-masks/{pellet}"
    np.save(savename, maskr)
    print(f"Saved mask into {savename}.")
