import numpy as np
from sklearn import neighbors

class ADASYN:
    def fit_resample(self, X, y, K=5, beta=1, threshold=1):
        seed = 997
        np.random.seed(seed)
        
        #Sprawdzanie liczby próbek klasy mniejszzościowej i większościowej
        ms = int(np.sum(y))
        ml = len(y) - ms

        clf = neighbors.KNeighborsClassifier()
        clf.fit(X, y)
        
        #Sprawdzanie czy dane są wystarczająco niezbalansowane
        d = np.divide(ms, ml)
        if d > threshold:
            return print("Niewystarczające niezbalansowanie zbioru danych")
        
        #Obliczanie całkowitej liczby próbek syntetycznych, ktore muszą zostać wygenerowane, aby osiągnąć poziom niezbalansowania beta
        G = (ml - ms) * beta
        
        Ri = []
        Minority_per_xi = []
        Minority_index = []
        
        #Określanie które numerów próbek mniejszościowych w puli wszystkich próbek
        for i in range(len(y)):
            if y[i] == 1:
                Minority_index.append(i)
                
        #Sprawdzanie najbliższych sąsiadów próbek klasy mniejszościowej        
        for i in range(len(y)):
            if y[i] == 1:
                xi = X[i, :].reshape(1, -1)
                neighbours = clf.kneighbors(xi, n_neighbors=K + 1, return_distance=False)[0]
                neighbours = neighbours[1:]
                count = 0
                #Liczenie ile sąsiadów próbki klasy mniejszościowej należy do klasy większościowej
                for value in neighbours:
                    if not value in Minority_index:
                        count += 1
                minority = []
                #Sprawdzanie sąsiadów mniejszościowych dla wszystkich próbek mniejszościowych i dodawanie ich do tablicy
                for value in neighbours:
                    if value in Minority_index:
                        minority.append(value)
                Minority_per_xi.append(minority)
                #Wyliczanie stopnia niezbalansowania każdego sąsiadztwa wszystkich próbek mniejszościowych (1 - brak sąsiadów mniejszościowych, 0 - wszyscy sąsiedzi są mniejszościowi)
                Ri.append(count / K)
        Rhat_i = []
        Ri_sum = sum(Ri)
        #Próbki mniejszościowe, które nie posiadają żadnych sąsiadów klasy mniejszościowej postanowiono zignorować. 
        #W związku z tym, postanowiono wykluczyć ich udział w wyliczaniu sumy Ri. Pozwoli to na uzyskanie balansu beta.
        for ri in Ri:
            if ri == 1:
                Ri_sum -= 1
        #Normalizujemy Ri. Proces ten wykonujemy również dla próbek które wcześniej wykluczyliśmy, jednak nie wpłynie to negatywnie na wyniki działania algorytmu.
        for ri in Ri:
            rhat_i = ri / Ri_sum
            Rhat_i.append(rhat_i)
        Gi = []
        #Wyliczamy liczbę próbek syntetycznych która ma powstać dla każdej próbki mniejszościowej.
        #Ponownie wyliczamy ją również dla wykluczonych próbek, jednak wyliczenia te nie będą brane pod uwagę w późniejszych etapach.
        for rhat_i in Rhat_i:
            gi = round(rhat_i * G)
            Gi.append(int(gi))
        syn_data = []
        
        #Sprawdzamy wszystkie próbki mniejszościowe. 
        #Dla próbek które mają sąsiadów mniejszościowych (tablica Minority_per_xi nie jest pusta) wykonywana jest generacja próbek syntetycznych.
        #Próbki które nie mają sąsiadów mniejszościowych są ignorowane.
        flag = 0
        for i in range(len(y)):
            if y[i] == 1:
                xi = X[i, :].reshape(1, -1)
                for j in range(Gi[flag]):
                    if Minority_per_xi[flag]:
                        index = np.random.choice(Minority_per_xi[flag])
                        xzi = X[index, :].reshape(1, -1)
                        si = xi + (xzi - xi) * np.random.uniform(0, 1)
                        syn_data.append(si)
                flag += 1
        data = []
        labels = []
        #Do tablicy data dodawane są próbki syntetyczne, a następnie łączymy ją ze starymi próbkami
        #Do tablicy labels dodawane są nowe etykiety, a następnie łączymy ją ze starymi etykietami
        for values in syn_data:
            data.append(values[0])
        labels = np.ones([len(data), 1])
        labels = np.concatenate([y.reshape(-1, 1), labels])
        data = np.concatenate([X, data])
        return data, labels
