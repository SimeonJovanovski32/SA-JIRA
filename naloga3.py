import cv2 as cv


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


def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

if __name__ == "__main__":
    pass
