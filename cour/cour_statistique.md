# 📚 Table des Matières – Formules de Statistique

## 1. Statistiques descriptives

### 1.1. Mesures de tendance centrale

#### • Moyenne arithmétique simple 

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- $x_i$  : les valeurs de la série
- $n$ : le nombre total de données

---

#### • Moyenne pondérée

$$\bar{x}_p = \frac{\sum_{i=1}^{n} x_i \cdot w_i}{\sum_{i=1}^{n} w_i}$$

- $x_i$ : les valeurs
- $w_i$ : les poids ou effectifs associés à chaque valeur

---

#### • Médiane

Si $n$ est impair :

$$M = x_{\frac{n+1}{2}}$$ 

Si $n$ est pair : 

$$M = \frac{x_{\frac{n}{2}} + x_{\frac{n}{2} + 1}}{2}$$  

- $M$ : Médiane (valeur centrale)
- $n$ : Nombre total de données
- $x_i$ : Valeurs triées de l'échantillon

---

#### • Mode

$$ \text{Mode} = \arg\max_x f(x)$$ 

- Mode : Valeur la plus fréquente dans l’échantillon
- $f(x)$ : Fréquence de $x$

---

### 1.2. Mesures de dispersion

#### • Étendue

$$\text{Étendue} = x_{\text{max}} - x_{\text{min}}$$

- $x_max$ : valeur maximale
- $x_min$ : valeur minimale

---

#### • Variance

$$s^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

- $x_i$ : les valeurs de la série
- $\bar{x}$ : la moyenne
- $n$ : nombre total de données

---

#### • Écart-type

$$\sigma = \sqrt{s^2}$$

- Racine carrée de la variance  
- Permet de mesurer l’**écart moyen à la moyenne**

---

#### • Écart interquartile

$$IQR = Q_3 - Q_1$$

- $Q_1$ : premier quartile (25 % des données)
- $Q_3$ : troisième quartile (75 % des données)

---

#### • Coefficient de variation


$$CV = \frac{\sigma}{\bar{x}} \times 100$$

- $σ$ : écart-type
- $\bar{x}$ : moyenne
- Exprimé en **%**, il permet de comparer la dispersion entre séries

---

### 1.3. Mesures de position

#### • Quartiles

- $Q1$ : 25 % des données sont inférieures ou égales à cette valeur  
- $Q2$ : médiane (50 %)  
- $Q3$ : 75 % des données sont inférieures ou égales à cette valeur
  
#### • Déciles

- $D1$ à $D9$ : divisent la série en 10 parties égales  
  Ex : $D4$ = 40 % des données ≤ D4

---

#### • Centiles

- $P1$ à $P99$ : divisent la série en 100 parties égales  
  Ex : $P90$ = 90 % des données ≤ P90

---

### 1.4. Tableaux statistiques

#### • Fréquence absolue

$$f_i = \text{nombre d’occurrences de la valeur } x_i$$

#### • Fréquence relative

$$f_i^{\text{rel}} = \frac{f_i}{n}$$

- $f_i$ : fréquence absolue
- $n$ : total des données

---

#### • Fréquence cumulée croissante

$$F_i = \sum_{j=1}^{i} f_j^{\text{rel}}$$

- Permet de connaître le pourcentage de données **inférieures ou égales** à une valeur donnée.

---

#### • Effectifs en classes (données groupées)

- On regroupe les valeurs dans des **intervalles** (ou classes)
- Pour chaque classe, on calcule :
  - **Effectif de classe** : nombre d’observations dans la classe
  - **Centre de classe** : $c = \frac{\text{borne inférieure + borne supérieure}}{2}$

---

## 2. Statistiques à deux variables

### 2.1. Covariance
#### Formule de la covariance  
#### Interprétation graphique  

### 2.2. Corrélation
#### Coefficient de corrélation linéaire de Pearson  
#### Valeurs possibles et interprétation  

### 2.3. Régression linéaire
#### Équation de la droite de régression : `y = ax + b`  
#### Calcul des coefficients `a` et `b`  
#### Coefficient de détermination `R²`

---

## 3. Variables aléatoires et espérance

### 3.1. Loi de probabilité discrète
#### Fonction de probabilité  
#### Espérance mathématique  
#### Variance et écart-type  

### 3.2. Lois usuelles
#### Loi uniforme discrète  
#### Loi de Bernoulli  
#### Loi binomiale

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- $P(X = k)$ : Probabilité d’obtenir exactement $k$ succès
- $n$ : Nombre d’essais
- $k$ : Nombre de succès souhaités
- $p$ : Probabilité de succès dans un essai
- $\binom{n}{k}$ : Coefficient binomial

---

#### Loi normale

$$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

- $f(x)$ : Fonction de densité de probabilité de la loi normale
- $x$ : Valeur de la variable aléatoire
- $\mu$ : Moyenne
- $\sigma$ : Écart-type

---

#### Loi de Poisson

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- $P(X = k)$ : Probabilité d’avoir exactement $k$ événements
- $\lambda$ : Taux moyen d'événements dans un intervalle
- $k$ : Nombre d'événements
- $k!$ : Factorielle de $k$

---

#### Loi géométrique

---

## 4. Estimation et intervalles

### 4.1. Estimation ponctuelle
#### Moyenne, proportion, variance d’un échantillon

### 4.2. Intervalle de confiance
#### Pour une moyenne (`σ` connue)  
#### Pour une proportion  
#### Taille d’un échantillon nécessaire

---

## 5. Tests statistiques (niveau avancé)

### 5.1. Hypothèses
#### Hypothèse nulle `H₀` et alternative `H₁`  
#### Risque d’erreur de type I et II

### 5.2. Test de moyenne / proportion
#### Z-test  
#### T-test (`σ` inconnue)

### 5.3. Khi²
#### Test d’indépendance  
#### Test d’ajustement

---

## 6. Compléments (niveau licence et plus)
#### Analyse de variance (ANOVA)  
#### Régression multiple  
#### Statistiques inférentielles  
#### Bootstrap (estimation par rééchantillonnage)



### **Coefficient de variation**  
$$CV = \frac{\sigma}{\bar{x}} \times 100\%$$  
- \( CV \) : Coefficient de variation
- \( \sigma \) : Écart-type
- \( \bar{x} \) : Moyenne de l'échantillon


## **2. Probabilités**

### **Probabilité d’un événement**  
$$P(A) = \frac{\text{nombre de cas favorables}}{\text{nombre de cas possibles}}$$  
- \( P(A) \) : Probabilité de l'événement \( A \)
- Nombre de cas favorables : Nombre de résultats où \( A \) se produit
- Nombre de cas possibles : Total des résultats possibles

### **Probabilité conditionnelle**  
$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$  
- \( P(A | B) \) : Probabilité de \( A \) sachant que \( B \) s'est produit
- \( P(A \cap B) \) : Probabilité que \( A \) et \( B \) se produisent
- \( P(B) \) : Probabilité de \( B \)

### **Théorème de Bayes**  
$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$  
- \( P(A | B) \) : Probabilité de \( A \) sachant \( B \)
- \( P(B | A) \) : Probabilité de \( B \) sachant \( A \)
- \( P(A) \) : Probabilité de \( A \)
- \( P(B) \) : Probabilité de \( B \)

---

## **3. Variables Aléatoires et Distributions**

### **Espérance mathématique**  
$$E(X) = \sum x_i P(x_i)$$  
- \( E(X) \) : Espérance de la variable aléatoire \( X \)
- \( x_i \) : Valeurs possibles de \( X \)
- \( P(x_i) \) : Probabilité associée à \( x_i \)

### **Variance d’une variable aléatoire**  
$$Var(X) = E(X^2) - (E(X))^2 $$  
- \( Var(X) \) : Variance de \( X \)
- \( E(X^2) \) : Espérance de \( X^2 \)
- \( E(X) \) : Espérance de \( X \)





## **4. Inférence Statistique**

### **Intervalle de confiance pour une moyenne**  
$$IC = \bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}} $$  
- \( IC \) : Intervalle de confiance pour la moyenne
- \( \bar{x} \) : Moyenne de l’échantillon
- \( z_{\alpha/2} \) : Valeur critique de la distribution normale
- \( \sigma \) : Écart-type de l’échantillon
- \( n \) : Taille de l’échantillon

### **Test d’hypothèse**  
- \( H_0 \) : Hypothèse nulle
- \( H_1 \) : Hypothèse alternative  
Utilise une statistique de test (calculée à partir des données) pour comparer avec un seuil critique afin de décider si \( H_0 \) est rejetée.

### **Test t de Student**  
$$ t = \frac{\bar{x} - \mu}{\frac{s}{\sqrt{n}}}$$  
- \( t \) : Statistique de test
- \( \bar{x} \) : Moyenne de l’échantillon
- \( \mu \) : Moyenne hypothétique de la population
- \( s \) : Écart-type de l’échantillon
- \( n \) : Taille de l’échantillon

### **Test du Chi-2**  
$$ \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$  
- \( \chi^2 \) : Statistique de test du chi-deux
- \( O_i \) : Observations (fréquences observées)
- \( E_i \) : Fréquences attendues

---

## **5. Régression et Corrélation**

### **Corrélation linéaire (coefficient de Pearson)**  
$$ r = \frac{\sum (x_i - \bar{x}) (y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$  
- \( r \) : Coefficient de corrélation de Pearson
- \( x_i \), \( y_i \) : Valeurs des variables \( X \) et \( Y \)
- \( \bar{x} \), \( \bar{y} \) : Moyennes respectives de \( X \) et \( Y \)

### **Régression linéaire simple**  
$$ y = ax + b $$  
- \( y \) : Variable dépendante
- \( x \) : Variable indépendante
- \( a \) : Pente de la régression
- \( b \) : Ordonnée à l'origine (intercept)

---

## **6. Statistiques Avancées et Applications**

### **Analyse en composantes principales (ACP)**  
$$Z = XW $$  
- \( Z \) : Matrice des composantes principales
- \( X \) : Matrice des données originales
- \( W \) : Matrice des vecteurs propres

### **Clustering (K-means)**  
$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$  
- \( J \) : Fonction de coût (distances intra-cluster)
- \( C_i \) : Cluster \( i \)
- \( \mu_i \) : Centre du cluster \( i \)
- \( x \) : Données dans le cluster

### **Régression logistique**  
$$P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$  
- \( P(Y=1 | X) \) : Probabilité que \( Y \) soit égal à 1 pour une valeur de \( X \)
- \( \beta_0 \), \( \beta_1 \) : Coefficients de régression
- \( X \) : Variable indépendante

### **Séries temporelles (ARIMA)**  
$$Y_t = \alpha + \sum \phi_i Y_{t-i} + \sum \theta_j \varepsilon_{t-j} + \varepsilon_t$$  
- \( Y_t \) : Valeur de la série temporelle à l'instant \( t \)
- \( \alpha \) : Constante
- \( \phi_i \) : Coefficients autorégressifs
- \( \varepsilon_t \) : Résidus (bruit aléatoire)
