# Manipulation et Nettoyage des Données en Python

## 📌 1. Charger et Analyser les Données avec Pandas

```python
import pandas as pd

# Charger un fichier CSV
df = pd.read_csv("data.csv")

# Afficher les 5 premières lignes
df.head()

# Obtenir des informations sur le DataFrame
df.info()

# Résumé statistique des données numériques
df.describe()
```

---

## 📌 2. Manipuler des Tableaux NumPy

```python
import numpy as np

# Créer un tableau NumPy
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Indexation et slicing
print(arr[0, 1])  # Accéder à l'élément en 1ère ligne, 2ème colonne
print(arr[:, 1])  # Sélectionner la 2ème colonne

# Opérations mathématiques
arr_squared = arr ** 2  # Élévation au carré
double_arr = arr * 2  # Multiplication par 2
```

---

## 📌 3. Gérer les Valeurs Manquantes et Aberrantes

```python
# Vérifier les valeurs manquantes
df.isnull().sum()

# Remplacer les valeurs manquantes par la moyenne
df.fillna(df.mean(), inplace=True)

# Supprimer les lignes contenant des valeurs manquantes
df.dropna(inplace=True)

# Détecter les valeurs aberrantes (outliers) avec l'IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
df_cleaned = df[~outliers.any(axis=1)]
```

---

## 📌 4. Effectuer des Jointures et Fusions de Datasets

```python
# Charger deux datasets
df1 = pd.DataFrame({"ID": [1, 2, 3], "Valeur": ["A", "B", "C"]})
df2 = pd.DataFrame({"ID": [2, 3, 4], "Score": [85, 90, 95]})

# Jointure sur la colonne 'ID'
merged_df = pd.merge(df1, df2, on="ID", how="inner")  # "left", "right" ou "outer"
```

---

## 📌 5. Encodage des Variables Catégorielles

```python
# One-Hot Encoding
df = pd.get_dummies(df, columns=["Categorie"], drop_first=True)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["Categorie"] = le.fit_transform(df["Categorie"])
```

---

## 📌 6. Normalisation et Standardisation des Données

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalisation (Min-Max Scaling)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Standardisation (Z-score Scaling)
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

---
# Programmation Orientée Objet (OOP) et Bonnes Pratiques en Python

## 📌 1. Définir des Classes et Objets

```python
# Définition d'une classe simple
class Voiture:
    def __init__(self, marque, modele, annee):
        self.marque = marque
        self.modele = modele
        self.annee = annee

    def afficher_infos(self):
        return f"{self.marque} {self.modele} - {self.annee}"

# Instanciation d'un objet
ma_voiture = Voiture("Tesla", "Model S", 2022)
print(ma_voiture.afficher_infos())  # Tesla Model S - 2022
```

---

## 📌 2. Héritage, Encapsulation et Polymorphisme

### 🔹 Héritage
```python
# Création d'une classe dérivée
class VoitureElectrique(Voiture):
    def __init__(self, marque, modele, annee, autonomie):
        super().__init__(marque, modele, annee)
        self.autonomie = autonomie

    def afficher_infos(self):
        return f"{super().afficher_infos()} - Autonomie: {self.autonomie} km"

# Instanciation d'un objet
ma_voiture_elec = VoitureElectrique("Tesla", "Model 3", 2023, 500)
print(ma_voiture_elec.afficher_infos())  # Tesla Model 3 - 2023 - Autonomie: 500 km
```

### 🔹 Encapsulation
```python
class CompteBancaire:
    def __init__(self, titulaire, solde):
        self.titulaire = titulaire
        self.__solde = solde  # Attribut privé

    def deposer(self, montant):
        self.__solde += montant

    def retirer(self, montant):
        if montant <= self.__solde:
            self.__solde -= montant
        else:
            print("Fonds insuffisants")

    def afficher_solde(self):
        return f"Solde de {self.titulaire}: {self.__solde} €"

# Utilisation
compte = CompteBancaire("Alice", 1000)
compte.deposer(500)
print(compte.afficher_solde())  # Solde de Alice: 1500 €
```

### 🔹 Polymorphisme
```python
class Animal:
    def parler(self):
        pass  # Méthode à implémenter

class Chien(Animal):
    def parler(self):
        return "Woof!"

class Chat(Animal):
    def parler(self):
        return "Miaou!"

# Fonction polymorphique
def faire_parler(animal):
    print(animal.parler())

# Utilisation
chien = Chien()
chat = Chat()
faire_parler(chien)  # Woof!
faire_parler(chat)   # Miaou!
```

---

## 📌 3. Utiliser les Design Patterns en Data Science

### 🔹 Singleton (Assurer une instance unique)
```python
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

# Vérification du Singleton
obj1 = Singleton()
obj2 = Singleton()
print(obj1 is obj2)  # True
```

### 🔹 Factory Pattern (Création flexible d'objets)
```python
class Factory:
    @staticmethod
    def creer_animal(type_animal):
        if type_animal == "chien":
            return Chien()
        elif type_animal == "chat":
            return Chat()
        else:
            raise ValueError("Type d'animal inconnu")

# Utilisation
animal = Factory.creer_animal("chien")
print(animal.parler())  # Woof!
```

---

## 📌  Structurer son Code pour le Rendre Réutilisable et Modulaire

### 🔹 Séparer le code en fichiers/modules
```
projet/
│── main.py  # Programme principal
│── models.py  # Contient les classes
│── utils.py  # Fonctions utilitaires
```

### 🔹 Exemple d'importation et de structuration
```python
# models.py
class Personne:
    def __init__(self, nom, age):
        self.nom = nom
        self.age = age
```

```python
# main.py
from models import Personne
p = Personne("Alice", 30)
print(p.nom)
```

# Optimisation et Structuration du Code en Python

## 📌 1. Utilisation des Fonctions Lambda, Map, Filter, List Comprehensions

```python
# Fonction lambda
carre = lambda x: x**2
print(carre(5))  # Résultat : 25

# Utilisation de map
nombres = [1, 2, 3, 4]
carres = list(map(lambda x: x**2, nombres))  # [1, 4, 9, 16]

# Utilisation de filter
pairs = list(filter(lambda x: x % 2 == 0, nombres))  # [2, 4]

# List comprehension
cubes = [x**3 for x in nombres]  # [1, 8, 27, 64]
```

---

## 📌 2. Utilisation Efficace des Boucles et Structures Conditionnelles

```python
# Mauvaise pratique : Boucle inefficace
result = []
for i in range(10):
    result.append(i**2)

# Bonne pratique : List comprehension
result = [i**2 for i in range(10)]
```

```python
# Utilisation des expressions ternaires
x = 10
type_nombre = "Pair" if x % 2 == 0 else "Impair"
```

---

## 📌 3. Éviter les Boucles Imbriquées avec NumPy et Pandas

```python
import numpy as np
import pandas as pd

# Mauvaise pratique : Boucle imbriquée
matrice = np.random.rand(1000, 1000)
somme = 0
for i in range(1000):
    for j in range(1000):
        somme += matrice[i, j]

# Bonne pratique : Utilisation de NumPy
somme = np.sum(matrice)
```

```python
# Utilisation de Pandas pour éviter les boucles
df = pd.DataFrame({"A": range(1, 6), "B": range(10, 15)})

df["Somme"] = df["A"] + df["B"]  # Évite une boucle explicite
```

---

## 📌 4. Profilage et Optimisation des Performances

```python
import cProfile

def calcul():
    return sum([i**2 for i in range(100000)])

cProfile.run('calcul()')
```

```python
from line_profiler import LineProfiler

def slow_function():
    result = 0
    for i in range(100000):
        result += i**2
    return result

profiler = LineProfiler()
profiler.add_function(slow_function)
profiler.enable()
slow_function()
profiler.disable()
profiler.print_stats()
```

---

## 📌 5. Parallélisation et Multiprocessing

```python
from joblib import Parallel, delayed
import multiprocessing

# Fonction à paralléliser
def carre(x):
    return x**2

n_jobs = multiprocessing.cpu_count()
resultats = Parallel(n_jobs=n_jobs)(delayed(carre)(i) for i in range(10))
print(resultats)
```

```python
from multiprocessing import Pool

def carre(x):
    return x**2

if __name__ == "__main__":
    with Pool(4) as p:
        print(p.map(carre, range(10)))
```

---

