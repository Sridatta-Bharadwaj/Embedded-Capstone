"""Remove ultrasound overlay artifacts and drop near-duplicate images (Phase 2).

Approach, decided after inspecting a 32-image sample grid of raw OTU_2D frames
(reports/figures/sample_grid_raw.png): frames are grayscale transvaginal-ultrasound fans on black, with
  * coloured calipers / dotted measurement lines / measurement text (yellow, green) inside the fan,
  * white-ish machine text and colour scale bars in the black corners outside the fan,
  * in some frames, colour-Doppler flow regions (large coloured blobs) inside the fan.
Cleaning therefore:
  1. inpaints SMALL coloured components only (calipers, lines, text) from surrounding tissue;
     large coloured regions (Doppler) are not inpainted — inpainting would smear real tissue — and are
     simply converted to grayscale like everything else;
  2. keeps only the scan fan: the largest bright connected region, filled via its convex hull; everything
     outside it (corner text, scale bars, colour bars) is set to black; then crops to the fan's bounding box.
  3. Near-duplicates: 64-bit perceptual hash (imagehash.phash) on the cleaned frame. Pairs within
     `dedupe_hamming` are near-identical -> keep one. Pairs within `group_hamming` are "visually similar"
     and are clustered into `sim_group`, which the split uses to keep similar frames (very likely the same
     patient / same scan session) in the same partition, since OTU_2D ships no patient IDs.

Only numpy/opencv at import time so the same cleaning runs on the Raspberry Pi.
"""
import cv2
import numpy as np

COLOR_SPREAD_THRESHOLD = 40     # max(R,G,B) - min(R,G,B) above this = coloured pixel
MAX_OVERLAY_COMPONENT = 0.004   # coloured components smaller than this fraction of the frame are overlay marks
CONTENT_THRESHOLD = 12          # gray level above which a pixel counts as scan content
MIN_CROP_FRACTION = 0.25        # if the detected fan is smaller than this, don't trust it
TABLE_MARGIN = 6                # px added around a detected measurement table before blacking it out


def overlay_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of small coloured overlay marks (not large Doppler regions)."""
    if bgr.ndim == 2:
        return np.zeros(bgr.shape, np.uint8)
    spread = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    colored = (spread > COLOR_SPREAD_THRESHOLD).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)
    max_area = MAX_OVERLAY_COMPONENT * colored.size
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] <= max_area
    mask = keep[labels].astype(np.uint8) * 255
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def remove_overlays(bgr: np.ndarray, keep_color: bool = False) -> tuple[np.ndarray, float]:
    """Inpaint small coloured overlays; returns (grayscale image — or BGR if keep_color — , fraction inpainted).
    keep_color=True preserves colour-Doppler flow (a vascularity cue) instead of flattening it to gray."""
    mask = overlay_mask(bgr)
    frac = float(mask.mean() / 255)
    if frac > 0:
        bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    if keep_color and bgr.ndim == 3:
        return bgr, frac
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return gray, frac


def fan_mask(gray: np.ndarray) -> np.ndarray | None:
    """Filled convex hull (uint8 0/1) of the largest non-black region = the scan fan; None if unreliable."""
    binary = (cv2.GaussianBlur(gray, (5, 5), 0) > CONTENT_THRESHOLD).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[i, :4])
    if w * h < MIN_CROP_FRACTION * gray.size:
        return None
    pts = cv2.findNonZero((labels == i).astype(np.uint8))
    mask = np.zeros_like(binary)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
    return mask


def table_rects(bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """(x, y, w, h) of on-screen measurement tables ("1 D 5.54cm / 2 D 3.62cm ..."): dense blocks of coloured
    text near the bottom of the frame on a flat dark background. Inpainting only removes their text and leaves a
    flat grey rectangle (a possible shortcut feature), so blackout_tables=True zeroes the whole rectangle.
    Scattered calipers / dotted lines are too sparse, and colour-Doppler regions too textured, to match."""
    if bgr.ndim == 2:
        return []
    H, W = bgr.shape[:2]
    spread = bgr.max(axis=2).astype(np.int16) - bgr.min(axis=2).astype(np.int16)
    colored = (spread > COLOR_SPREAD_THRESHOLD).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n, _, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(colored, np.ones((7, 15), np.uint8)), connectivity=8)
    rects = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if w < 0.06 * W or h < 0.04 * H or y + h < 0.8 * H:
            continue
        box = colored[y:y + h, x:x + w]
        density, background = box.mean(), gray[y:y + h, x:x + w][box == 0]
        if 0.1 <= density <= 0.6 and background.size and background.std() <= 25:
            m = TABLE_MARGIN
            x0, y0 = max(0, x - m), max(0, y - m)
            rects.append((x0, y0, min(W, x + w + m) - x0, min(H, y + h + m) - y0))
    return rects


def clean_image(bgr: np.ndarray, keep_color: bool = False, blackout_tables: bool = False) -> tuple[np.ndarray, dict]:
    """Full cleaning for one frame -> (cropped uint8 image — grayscale, or BGR if keep_color — , info dict).
    blackout_tables=True additionally sets measurement-table rectangles to black (after the fan crop is found,
    so the crop geometry is identical with and without it)."""
    rects = table_rects(bgr) if blackout_tables else []
    img, overlay_frac = remove_overlays(bgr, keep_color)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = fan_mask(gray)
    if mask is None:
        h, w = gray.shape
        for tx, ty, tw, th in rects:
            img[ty:ty + th, tx:tx + tw] = 0
        return img, {"tables_blacked": len(rects), "crop_x": 0, "crop_y": 0, "crop_w": w, "crop_h": h,
                     "overlay_frac": round(overlay_frac, 5), "fan_found": False}
    img = img * (mask[:, :, None] if img.ndim == 3 else mask)
    for tx, ty, tw, th in rects:
        img[ty:ty + th, tx:tx + tw] = 0
    x, y, w, h = cv2.boundingRect(mask)
    return img[y:y + h, x:x + w], {"crop_x": x, "crop_y": y, "crop_w": w, "crop_h": h,
                                   "overlay_frac": round(overlay_frac, 5), "fan_found": True,
                                   "tables_blacked": len(rects)}


# ---------------------------------------------------------------- near-duplicate detection
def phash_bits(gray: np.ndarray) -> np.ndarray:
    """64-bit perceptual hash as a bool vector."""
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.fromarray(gray), hash_size=8).hash.flatten()


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between bit vectors (memory ~ n^2 int16)."""
    packed = np.packbits(bits.astype(bool), axis=1)
    dist = np.zeros((len(packed), len(packed)), np.int16)
    for k in range(packed.shape[1]):
        dist += _POPCOUNT[packed[:, None, k] ^ packed[None, :, k]]
    return dist


def _components(adj: np.ndarray) -> np.ndarray:
    """Connected-component labels of a boolean adjacency matrix (union-find)."""
    n = len(adj)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(adj, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([remap[r] for r in roots])


def duplicate_groups(bits: np.ndarray, dedupe_hamming: int, group_hamming: int):
    """Returns (dup_cluster, sim_group) label arrays."""
    dist = hamming_matrix(bits)
    return _components(dist <= dedupe_hamming), _components(dist <= group_hamming)
