import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''

    # Pretvorimo sliko v 2D tabelo, kjer vsaka vrstica predstavlja en piksel in njegove RGB vrednosti
    slika_reshape = slika.reshape((-1, 3))

    # Inicializiramo k centroids na naključnih vrednostih v območju slikovnih pikslov
    centroids = slika_reshape[np.random.choice(slika_reshape.shape[0], k, replace=False)]

    for _ in range(iteracije):
        # Izračunaj razdaljo od vseh pikslov do vseh centrov (Evklidska razdalja)
        distances = np.linalg.norm(slika_reshape[:, np.newaxis] - centroids, axis=2)

        # Dodelimo vsak piksel najbližjemu centru
        labels = np.argmin(distances, axis=1)

        # Izračunamo nove centre kot povprečje vseh pikslov, dodeljenih vsakemu centru
        new_centroids = np.array([slika_reshape[labels == i].mean(axis=0) for i in range(k)])

        # Preverimo, če so se centroids spremenili
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Zdaj imamo dodeljene oznake za vsak piksel, spremenimo sliko tako, da bo imela barve centrov
    segmented_image = centroids[labels].reshape(slika.shape)

    return np.uint8(segmented_image)
    
def meanshift(slika, h, dimenzija):
    visina, sirina, _ = slika.shape
    num_features = slika.shape[2]
    slika_reshape = slika.reshape((-1, num_features))

    if dimenzija == 5:
        coordinates = np.column_stack(np.indices((visina, sirina)))
        slika_reshape = np.hstack([slika_reshape, coordinates.reshape(-1, 2)])

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
            utezi_sum = np.sum(utezi)
            nova_tocka = np.sum(utezi[:, np.newaxis] * slika_reshape, axis=0) / utezi_sum
            iteracija += 1

        converged_points.append(nova_tocka)

    centers = []
    for point in converged_points:
        close_centers = [center for center in centers if np.linalg.norm(center - point) < min_cd]
        if close_centers:
            closest_center = min(close_centers, key=lambda c: np.linalg.norm(c - point))
            centers[centers.index(closest_center)] = (np.array(closest_center) + point) / 2
        else:
            centers.append(point)

    centers = np.array(centers)
    labels = np.argmin(np.linalg.norm(slika_reshape[:, np.newaxis] - centers, axis=2), axis=1)
    segmented_image = centers[labels].reshape(slika.shape)
    return np.uint8(segmented_image)
    
def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    # Pridobimo dimenzije slike
    visina, sirina, _ = slika.shape

    # Pretvorimo sliko v 2D tabelo, kjer vsaka vrstica predstavlja en piksel in njegove RGB vrednosti
    slika_reshape = slika.reshape((-1, 3))

    if dimenzija_centra == 5:
        # Dodamo koordinate (X, Y) vsakega piksla
        coordinates = np.column_stack(np.indices((visina, sirina)))
        slika_reshape = np.hstack([slika_reshape, coordinates.reshape(-1, 2)])

    # Izbira centrov
    if izbira == 'ročno':
        # Tukaj je potrebna implementacija za ročno izbiro centrov (s klikom na sliko)
        # Predpostavljamo, da imate neko orodje za to, ki vrača seznam izbranih centrov.
        print("Ročna izbira centrov je potrebna")
        return []

    elif izbira == 'naključna':
        # Naključna izbira centrov
        k = 3  # Ali prilagodite k, kako ga določiti
        centroids = []

        # Naključno izberemo k začetnih centrov, preverimo razdalje med njimi
        while len(centroids) < k:
            # Naključen indeks piksla
            index = np.random.choice(slika_reshape.shape[0])
            new_centroid = slika_reshape[index]

            # Preverimo, če so centri dovolj oddaljeni
            if len(centroids) == 0:
                centroids.append(new_centroid)
            else:
                distances = np.linalg.norm(np.array(centroids)[:, :dimenzija_centra] - new_centroid[:dimenzija_centra], axis=1)
                if np.all(distances > T):
                    centroids.append(new_centroid)

        return np.array(centroids)
    



if __name__ == "__main__":
    # Load the image
    slika = cv.imread('zelenjave.jpg')  # Replace with the path to your image
    if slika is None:
        print("Error: Image not found.")
    else:
        # Convert the image to RGB (OpenCV loads images in BGR format by default)
        slika_rgb = cv.cvtColor(slika, cv.COLOR_BGR2RGB)

        # Apply the k-means function
        k = 3  # Number of clusters
        iteracije = 10  # Number of iterations
        segmented_image = kmeans(slika_rgb, k=k, iteracije=iteracije)

        # Display the original and segmented image using matplotlib
        plt.figure(figsize=(10, 5))

        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(slika_rgb)
        plt.title("Original Image")
        plt.axis('off')

        # Segmented image
        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.title("Segmented Image")
        plt.axis('off')

        plt.show()

        # Optionally save the segmented image
        segmented_image_bgr = cv.cvtColor(segmented_image, cv.COLOR_RGB2BGR)
        cv.imwrite('segmented_zelenjave_sliko.jpg', segmented_image_bgr)

        # Apply the Mean-Shift function with different settings
        meanshift_segmented_image_h10 = meanshift(slika_rgb, h=10, dimenzija=3)
        meanshift_segmented_image_h30 = meanshift(slika_rgb, h=30, dimenzija=3)

        # Prepare features for dimenzija=5
        visina, sirina, _ = slika_rgb.shape
        coordinates = np.column_stack(np.indices((visina, sirina)))
        slika_features = np.hstack([slika_rgb.reshape(-1, 3), coordinates.reshape(-1, 2)]).reshape((visina, sirina, 5))

        meanshift_segmented_image_dim5 = meanshift(slika_features, h=20, dimenzija=5)

        # Display results
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(slika_rgb)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(segmented_image)
        plt.title("K-means (k=3)")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(meanshift_segmented_image_h10)
        plt.title("Mean-Shift (h=10, dim=3)")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(meanshift_segmented_image_h30)
        plt.title("Mean-Shift (h=30, dim=3)")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(meanshift_segmented_image_dim5)
        plt.title("Mean-Shift (h=20, dim=5)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Save segmented images
        cv.imwrite('meanshift_h10.jpg', cv.cvtColor(meanshift_segmented_image_h10, cv.COLOR_RGB2BGR))
        cv.imwrite('meanshift_h30.jpg', cv.cvtColor(meanshift_segmented_image_h30, cv.COLOR_RGB2BGR))
        cv.imwrite('meanshift_dim5.jpg', cv.cvtColor(meanshift_segmented_image_dim5, cv.COLOR_RGB2BGR))

        # Demonstration: when does it make sense to use location (dim=5)?
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(meanshift_segmented_image_h30)
        plt.title("Mean-Shift without location (h=30, dim=3)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(meanshift_segmented_image_dim5)
        plt.title("Mean-Shift with location (h=20, dim=5)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
