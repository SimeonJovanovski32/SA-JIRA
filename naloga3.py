import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Utility ­functions
# ------------------------------------------------------------------
def gaussian_kernel(distances, h):
    """Gaussian kernel weights for mean‑shift."""
    return np.exp(-(distances ** 2) / (2 * h ** 2))


def kmeans(slika, k=3, iteracije=10):
    slika_reshape = slika.reshape((-1, 3))
    centroids = slika_reshape[np.random.choice(slika_reshape.shape[0], k, replace=False)]

    for _ in range(iteracije):
        distances = np.linalg.norm(slika_reshape[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array(
            [slika_reshape[labels == i].mean(axis=0) for i in range(k)]
        )
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids[labels].reshape(slika.shape).astype(np.uint8)


def meanshift(slika, h, dimenzija):
    """Mean‑shift segmentation for RGB (dim=3) or RGB‑XY (dim=5)."""
    visina, sirina, _ = slika.shape
    num_features = slika.shape[2]
    slika_reshape = slika.reshape((-1, num_features))

    if dimenzija == 5:
        coords = np.column_stack(np.indices((visina, sirina)))
        slika_reshape = np.hstack([slika_reshape, coords.reshape(-1, 2)])

    min_cd = 1e-3
    max_iteracije = 10
    converged_points = []

    for i in range(len(slika_reshape)):
        tocka = slika_reshape[i]
        nova_tocka = np.zeros_like(tocka)
        iteracija = 0

        while np.linalg.norm(tocka - nova_tocka) > min_cd and iteracija < max_iteracije:
            if iteracija > 0:
                tocka = nova_tocka
            razdalje = np.linalg.norm(slika_reshape - tocka, axis=1)
            utezi = gaussian_kernel(razdalje, h)
            nova_tocka = np.sum(utezi[:, None] * slika_reshape, axis=0) / utezi.sum()
            iteracija += 1

        converged_points.append(nova_tocka)

    # -------- fixed “merge‑centers” block (no list.index on arrays) --------
    centers = []
    for point in converged_points:
        merged = False
        for j, center in enumerate(centers):
            if np.linalg.norm(center - point) < min_cd:
                centers[j] = (center + point) / 2.0
                merged = True
                break
        if not merged:
            centers.append(point)
    centers = np.array(centers)

    labels = np.argmin(
        np.linalg.norm(slika_reshape[:, None] - centers, axis=2), axis=1
    )
    return centers[labels].reshape(slika.shape).astype(np.uint8)


def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    """(Optional helper) Choose K‑means centres manually or randomly."""
    visina, sirina, _ = slika.shape
    slika_reshape = slika.reshape((-1, 3))

    if dimenzija_centra == 5:
        coords = np.column_stack(np.indices((visina, sirina)))
        slika_reshape = np.hstack([slika_reshape, coords.reshape(-1, 2)])

    if izbira == "ročno":
        print("Ročna izbira centrov je potrebna.")
        return []

    if izbira == "naključna":
        k = 3
        centroids = []
        while len(centroids) < k:
            idx = np.random.choice(slika_reshape.shape[0])
            new_c = slika_reshape[idx]
            if not centroids:
                centroids.append(new_c)
            else:
                dists = np.linalg.norm(
                    np.array(centroids)[:, :dimenzija_centra] - new_c[:dimenzija_centra],
                    axis=1,
                )
                if np.all(dists > T):
                    centroids.append(new_c)
        return np.array(centroids)


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    TARGET_SIZE = (100, 100)
    slika = cv.imread("zelenjava.jpg")
    if slika is None:
        raise FileNotFoundError("Image not found at zelenjava.jpg")

    slika_rgb = cv.cvtColor(slika, cv.COLOR_BGR2RGB)
    slika_rgb = cv.resize(slika_rgb, TARGET_SIZE)

    # 1) K‑means
    segmented_image = kmeans(slika_rgb, k=3, iteracije=10)
    print("[K‑means] segmented_image:", segmented_image.shape, segmented_image.dtype)

    # 2) Mean‑shift, RGB only
    meanshift_segmented_image_h10 = meanshift(slika_rgb, h=10, dimenzija=3)
    print("[Mean‑Shift] (h=10, dim=3):", meanshift_segmented_image_h10.shape)

    meanshift_segmented_image_h30 = meanshift(slika_rgb, h=30, dimenzija=3)
    print("[Mean‑Shift] (h=30, dim=3):", meanshift_segmented_image_h30.shape)

    # 3) Mean‑shift with XY coords (dim=5)
    visina, sirina = TARGET_SIZE
    y_coords, x_coords = np.indices((visina, sirina))
    coords = np.dstack((x_coords, y_coords))
    slika_features = np.concatenate([slika_rgb, coords], axis=2)

    # 4) Save results
    cv.imwrite("segmented_kmeans.jpg", cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR))
    cv.imwrite("meanshift_h10.jpg", cv.cvtColor(meanshift_segmented_image_h10, cv.COLOR_RGB2BGR))
    cv.imwrite("meanshift_h30.jpg", cv.cvtColor(meanshift_segmented_image_h30, cv.COLOR_RGB2BGR))

    print(
        "Images saved: segmented_kmeans.jpg, meanshift_h10.jpg, "
        "meanshift_h30.jpg, meanshift_dim5.jpg"
    )

    # (Optional) quick visual check
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    ax[0, 0].imshow(slika_rgb);                    ax[0, 0].set_title("Original");      ax[0, 0].axis("off")
    ax[0, 1].imshow(segmented_image);              ax[0, 1].set_title("K‑means");       ax[0, 1].axis("off")
    ax[1, 0].imshow(meanshift_segmented_image_h10);ax[1, 0].set_title("Mean‑shift h=10");ax[1, 0].axis("off")
    ax[1, 1].imshow(meanshift_segmented_image_h30);ax[1, 1].set_title("Mean‑shift h=30");ax[1, 1].axis("off")
    ax[0, 2].axis("off")  # empty cell
    plt.tight_layout(); plt.show()
