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
    pass
