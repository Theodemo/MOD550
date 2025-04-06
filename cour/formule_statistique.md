<!-- /TOC -->
- [Formules de Statistique](#formules-de-statistique)
  - [1. Statistiques descriptives](#1-statistiques-descriptives)
    - [1.1 Mesures de tendance centrale](#11-mesures-de-tendance-centrale)
      - [Moyenne arithmétique simple](#moyenne-arithmétique-simple)
      - [Moyenne pondérée](#moyenne-pondérée)
      - [Moyenne géométrique](#moyenne-géométrique)
      - [Moyenne harmonique](#moyenne-harmonique)
      - [Médiane](#médiane)
      - [Mode](#mode)
    - [1.2 Mesures de dispersion](#12-mesures-de-dispersion)
      - [• Étendue](#-étendue)
      - [• Variance](#-variance)
      - [• Écart-type](#-écart-type)
      - [• Écart interquartile (IQR)](#-écart-interquartile-iqr)
      - [Coefficient de variation](#coefficient-de-variation)
    - [1.3 Mesures de position](#13-mesures-de-position)
      - [Quartiles](#quartiles)
      - [Percentiles](#percentiles)
      - [Déciles](#déciles)
    - [1.4 Représentations graphiques](#14-représentations-graphiques)
      - [Histogramme](#histogramme)
      - [Diagramme en boîte (boxplot)](#diagramme-en-boîte-boxplot)
      - [Nuage de points](#nuage-de-points)
      - [Diagramme circulaire](#diagramme-circulaire)
      - [Barres, courbes](#barres-courbes)
      - [Coefficient de corrélation linéaire de Pearson](#coefficient-de-corrélation-linéaire-de-pearson)
      - [Valeurs possibles et interprétation](#valeurs-possibles-et-interprétation)
  - [2. Régression et Corrélation](#2-régression-et-corrélation)
    - [2.1 Régression linéaire](#21-régression-linéaire)
      - [Équation de la droite de régression](#équation-de-la-droite-de-régression)
      - [Calcul des coefficients $a$ et $b$](#calcul-des-coefficients-a-et-b)
      - [Coefficient de détermination $R^2$](#coefficient-de-détermination-r2)
  - [3. Variables aléatoires et espérance](#3-variables-aléatoires-et-espérance)
    - [3.1. Loi de probabilité discrète](#31-loi-de-probabilité-discrète)
      - [Fonction de probabilité](#fonction-de-probabilité)
      - [Espérance mathématique](#espérance-mathématique)
      - [Variance et écart-type](#variance-et-écart-type)
    - [3.2. Lois usuelles](#32-lois-usuelles)
      - [Loi uniforme discrète](#loi-uniforme-discrète)
      - [Loi de Bernoulli](#loi-de-bernoulli)
      - [Loi binomiale](#loi-binomiale)
      - [Loi normale](#loi-normale)
      - [Loi de Poisson](#loi-de-poisson)
    - [Loi géométrique](#loi-géométrique)
      - [Formule de probabilité](#formule-de-probabilité)
      - [Espérance et variance](#espérance-et-variance)
  - [4. Estimation et intervalles](#4-estimation-et-intervalles)
    - [4.1. Estimation ponctuelle](#41-estimation-ponctuelle)
      - [Moyenne, proportion, variance d’un échantillon](#moyenne-proportion-variance-dun-échantillon)
    - [4.2. Intervalle de confiance](#42-intervalle-de-confiance)
      - [Pour une moyenne (`σ` connue)](#pour-une-moyenne-σ-connue)
      - [Pour une proportion](#pour-une-proportion)
      - [Taille d’un échantillon nécessaire](#taille-dun-échantillon-nécessaire)
  - [5. Tests statistiques (niveau avancé)](#5-tests-statistiques-niveau-avancé)
    - [5.1. Hypothèses](#51-hypothèses)
      - [Hypothèse nulle `H₀` et alternative `H₁`](#hypothèse-nulle-h-et-alternative-h)
      - [Risque d’erreur de type I et II](#risque-derreur-de-type-i-et-ii)
    - [5.2. Test de moyenne / proportion](#52-test-de-moyenne--proportion)
      - [Z-test (si $\\sigma$ connue)](#z-test-si-sigma-connue)
      - [T-test (si $\\sigma$ inconnue)](#t-test-si-sigma-inconnue)
    - [5.3. Khi²](#53-khi)
      - [Test d’indépendance](#test-dindépendance)
      - [Test d’ajustement](#test-dajustement)
  - [6. Compléments](#6-compléments)
    - [Analyse de variance (ANOVA)](#analyse-de-variance-anova)
    - [Régression multiple](#régression-multiple)
    - [Statistiques inférentielles](#statistiques-inférentielles)
    - [Bootstrap (estimation par rééchantillonnage)](#bootstrap-estimation-par-rééchantillonnage)
    - [Régression logistique](#régression-logistique)
    - [Analyse en composantes principales (ACP)](#analyse-en-composantes-principales-acp)
    - [Clustering (K-means)](#clustering-k-means)
    - [Séries temporelles (ARIMA)](#séries-temporelles-arima)
  - [7. Probabilités et Théorèmes](#7-probabilités-et-théorèmes)
    - [Probabilité d’un événement](#probabilité-dun-événement)
    - [Probabilité conditionnelle](#probabilité-conditionnelle)
    - [Théorème de Bayes](#théorème-de-bayes)
  - [8. Variables Aléatoires et Distributions](#8-variables-aléatoires-et-distributions)
    - [Espérance de la variable aléatoire](#espérance-de-la-variable-aléatoire)
    - [Variance d’une variable aléatoire](#variance-dune-variable-aléatoire)
  - [9. Inférence Statistique](#9-inférence-statistique)
    - [Intervalle de confiance pour une moyenne](#intervalle-de-confiance-pour-une-moyenne)
    - [Test d’hypothèse](#test-dhypothèse)
    - [Test t de Student](#test-t-de-student)
    - [Test du Chi-2](#test-du-chi-2)

<!-- /TOC -->

# Formules de Statistique

## 1. Statistiques descriptives

### 1.1 Mesures de tendance centrale

#### Moyenne arithmétique simple

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- $x_i$  : les valeurs de la série
- $n$ : le nombre total de données

---

#### Moyenne pondérée

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i \cdot w_i}{\sum_{i=1}^{n} w_i}$$

- $x_i$ : les valeurs
- $w_i$ : les poids ou effectifs associés à chaque valeur

---

#### Moyenne géométrique

#### Moyenne harmonique

#### Médiane

Si $n$ est impair :

$$M = x_{\frac{n+1}{2}}$$

Si $n$ est pair :

$$M = \frac{x_{\frac{n}{2}} + x_{\frac{n}{2} + 1}}{2}$$  

- $M$ : Médiane (valeur centrale)
- $n$ : Nombre total de données
- $x_i$ : Valeurs triées de l'échantillon

---

#### Mode

$$ \text{Mode} = \arg\max_x f(x)$$

- Mode : Valeur la plus fréquente dans l’échantillon
- $f(x)$ : Fréquence de $x$

---

### 1.2 Mesures de dispersion

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

#### • Écart interquartile (IQR)

$$IQR = Q_3 - Q_1$$

- $Q_1$ : premier quartile (25 % des données)
- $Q_3$ : troisième quartile (75 % des données)

---

#### Coefficient de variation

$$CV = \frac{\sigma}{\bar{x}} \times 100\%$$

- $CV$ : Coefficient de variation
- $\sigma$ : Écart-type
- $\bar{x}$ : Moyenne de l'échantillon

---

### 1.3 Mesures de position

#### Quartiles

#### Percentiles

#### Déciles

### 1.4 Représentations graphiques

#### Histogramme

#### Diagramme en boîte (boxplot)

#### Nuage de points

#### Diagramme circulaire

#### Barres, courbes

#### Coefficient de corrélation linéaire de Pearson

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

où :

- $r$ est le coefficient de corrélation de Pearson.
- $\text{Cov}(X, Y)$ est la covariance entre les variables $X$ et $Y$.
- $\sigma_X$ est l'écart-type de $X$.
- $\sigma_Y$ est l'écart-type de $Y$.

---

$$ r = \frac{\sum (x_i - \bar{x}) (y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$

- $r$ : Coefficient de corrélation de Pearson
- $x_i$, $y_i$ : Valeurs des variables $X$ et $Y$
- $\bar{x}$, $\bar{y}$: Moyennes respectives de $X $et $Y$

---

#### Valeurs possibles et interprétation

- $r = 1$ : Corrélation parfaite et positive.
- $r = -1$ : Corrélation parfaite et négative.
- $r = 0$ : Aucune corrélation linéaire.
- $0 < r < 1$ : Corrélation positive (les variables augmentent ensemble).
- $-1 < r < 0$ : Corrélation négative (l'une des variables augmente tandis que l'autre diminue).

Plus la valeur absolue de $r$ est proche de 1, plus la relation linéaire entre les deux variables est forte.

---

## 2. Régression et Corrélation

### 2.1 Régression linéaire

La **régression linéaire** permet de modéliser la relation entre une variable dépendante $Y$ et une variable indépendante $X$. Elle donne une équation de droite qui prédit $Y$ à partir de $X$.

#### Équation de la droite de régression

$$y = ax + b$$

où :

- $y$ est la variable dépendante (la variable que l'on cherche à prédire).
- $x$ est la variable indépendante (la variable utilisée pour prédire $y$).
- $a$ est le **coefficient directeur** ou pente de la droite, qui représente la variation de $y$ par unité de $x$.
- $b$ est l'**ordonnée à l'origine**, qui représente la valeur de $y$ lorsque $x = 0$.

---

#### Calcul des coefficients $a$ et $b$

- La pente $a$ est donnée par :
  
  $$a = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$

  où $\text{Cov}(X, Y)$ est la covariance et $\text{Var}(X)$ est la variance de $X$.
  
- L'ordonnée à l'origine $b$ est donnée par :
  
  $$b = \overline{Y} - a \overline{X}$$

  où $\overline{Y}$ et $\overline{X}$ sont les moyennes des variables $Y$ et $X$, respectivement.

---

#### Coefficient de détermination $R^2$

Le **coefficient de détermination** $R^2$ mesure la proportion de la variance de $Y$ qui est expliquée par la régression sur $X$.

$$R^2 = \frac{\text{Cov}^2(X, Y)}{\text{Var}(X) \cdot \text{Var}(Y)}$$

où :

- $R^2$ varie entre 0 et 1.
- $R^2 = 1$ indique que le modèle de régression explique parfaitement les données.
- $R^2 = 0$ indique que le modèle de régression n'explique rien de la variance de $Y$.

En résumé, plus $R^2$ est proche de 1, plus le modèle de régression est efficace pour expliquer la relation entre $X$ et $Y$.

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

### Loi géométrique

La **loi géométrique** modélise le **nombre d’essais nécessaires avant le premier succès** dans une série d’épreuves de Bernoulli indépendantes.

#### Formule de probabilité

$$P(X = k) = (1 - p)^{k - 1} \cdot p$$

où :

- $X$ est le nombre d’essais jusqu’au premier succès,
- $k \geq 1$ est un entier,
- $p$ est la probabilité de succès à chaque essai.

---

#### Espérance et variance

$$E(X) = \frac{1}{p}, \quad V(X) = \frac{1 - p}{p^2}$$

---

## 4. Estimation et intervalles

### 4.1. Estimation ponctuelle

#### Moyenne, proportion, variance d’un échantillon

- **Moyenne échantillonnale** :
  
  $$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- **Proportion échantillonnale** :
  
  $$\hat{p} = \frac{\text{nombre de succès}}{n}$$

- **Variance échantillonnale** :
  
  $$s^2 = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

où :

- $n$ est la taille de l’échantillon,
- $x_i$ sont les valeurs de l’échantillon,
- $\bar{x}$ est la moyenne de l’échantillon,
- $\hat{p}$ est la proportion estimée.

---

### 4.2. Intervalle de confiance

#### Pour une moyenne (`σ` connue)

$$IC = \left[ \bar{x} - z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}},\ \bar{x} + z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \right]$$

où :

- $\bar{x}$ est la moyenne de l’échantillon,
- $\sigma$ est l’écart-type de la population,
- $n$ est la taille de l’échantillon,
- $z_{\alpha/2}$ est la valeur critique de la loi normale.

#### Pour une proportion

$$IC = \left[ \hat{p} - z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}},\ \hat{p} + z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}} \right]$$

où :

- $\hat{p}$ est la proportion estimée,
- $n$ est la taille de l’échantillon,
- $z_{\alpha/2}$ est la valeur critique.

#### Taille d’un échantillon nécessaire

Pour une proportion :

$$n = \frac{z_{\alpha/2}^2 \cdot \hat{p}(1 - \hat{p})}{E^2}$$

Pour une moyenne :

$$n = \left( \frac{z_{\alpha/2} \cdot \sigma}{E} \right)^2$$

où :

- $E$ est la **marge d’erreur souhaitée**.

---

## 5. Tests statistiques (niveau avancé)

### 5.1. Hypothèses

#### Hypothèse nulle `H₀` et alternative `H₁`

- **H₀** : hypothèse qu'on teste (ex : $\mu = \mu_0$)
- **H₁** : hypothèse alternative (ex : $\mu \ne \mu_0$, $\mu > \mu_0$, ou $\mu < \mu_0$)

#### Risque d’erreur de type I et II

- **Erreur de type I** : rejeter H₀ alors qu’elle est vraie (probabilité = $\alpha$)
- **Erreur de type II** : ne pas rejeter H₀ alors qu’elle est fausse (probabilité = $\beta$)

---

### 5.2. Test de moyenne / proportion

#### Z-test (si $\sigma$ connue)

$$z = \frac{\bar{x} - \mu_0}{\frac{\sigma}{\sqrt{n}}}$$

où :

- $\bar{x}$ est la moyenne observée,
- $\mu_0$ est la moyenne théorique,
- $\sigma$ est l’écart-type connu,
- $n$ est la taille de l’échantillon.

#### T-test (si $\sigma$ inconnue)

$$t = \frac{\bar{x} - \mu_0}{\frac{s}{\sqrt{n}}}$$

où :

- $s$ est l’écart-type de l’échantillon,
- Les autres variables sont identiques au Z-test.
- La statistique suit une loi de Student à $n - 1$ degrés de liberté.

---

### 5.3. Khi²

#### Test d’indépendance

$$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

où :

- $O_{ij}$ est l’effectif observé dans la cellule $i, j$,
- $E_{ij}$ est l’effectif attendu sous H₀.

#### Test d’ajustement

Même formule que le test d’indépendance, mais appliqué à un seul caractère pour vérifier l’adéquation à une loi théorique (loi uniforme, binomiale, etc.).

---

## 6. Compléments

### Analyse de variance (ANOVA)

Permet de comparer les moyennes de plusieurs groupes :

$$F = \frac{\text{variance inter-groupe}}{\text{variance intra-groupe}}$$

Si $F$ est grand, les groupes sont probablement différents.

### Régression multiple

Extension de la régression linéaire à plusieurs variables indépendantes :

$$y = a_1x_1 + a_2x_2 + \dots + a_kx_k + b$$

où :

- $x_1, x_2, \dots$ sont les variables explicatives.

### Statistiques inférentielles

C’est l’ensemble des méthodes permettant de **faire des généralisations** ou des tests sur une population à partir d’un échantillon :

- Estimations
- Tests d’hypothèse
- Intervalles de confiance

### Bootstrap (estimation par rééchantillonnage)

Méthode de simulation pour approximer une distribution d’un estimateur :

- Tirages aléatoires avec remise dans l’échantillon initial
- Permet de calculer une estimation de l’écart-type, moyenne, intervalle de confiance même sans connaître la loi sous-jacente

---

### Régression logistique

$$P(Y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$

- $P(Y=1 | X)$ : Probabilité que $Y$ soit égal à 1 pour une valeur de $X$
- $\beta_0$, $\beta_1$ : Coefficients de régression
- $X$ : Variable indépendante

---

### Analyse en composantes principales (ACP)

$$Z = XW $$

- $Z$ : Matrice des composantes principales
- $X$ : Matrice des données originales
- $W$ : Matrice des vecteurs propres

---

### Clustering (K-means)

$$J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2$$

- $J$ : Fonction de coût (distances intra-cluster)
- $C_i$ : Cluster $i$
- $\mu_i$ : Centre du cluster $i$
- $x$ : Données dans le cluster

---

### Séries temporelles (ARIMA)

$$Y_t = \alpha + \sum \phi_i Y_{t-i} + \sum \theta_j \varepsilon_{t-j} + \varepsilon_t$$

- $Y_t$ : Valeur de la série temporelle à l'instant $t$
- $\alpha$ : Constante
- $\phi_i$ : Coefficients autorégressifs
- $\varepsilon_t$ : Résidus (bruit aléatoire)

---

## 7. Probabilités et Théorèmes

### Probabilité d’un événement

$$P(A) = \frac{\text{nombre de cas favorables}}{\text{nombre de cas possibles}}$$

- $P(A)$ : Probabilité de l'événement $A$
- Nombre de cas favorables : Nombre de résultats où $A$ se produit
- Nombre de cas possibles : Total des résultats possibles

---

### Probabilité conditionnelle

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

- $P(A | B)$ : Probabilité de $A$ sachant que $B$ s'est produit
- $P(A \cap B)$ : Probabilité que $A$ et $B$ se produisent
- $P(B)$ : Probabilité de $B$

---

### Théorème de Bayes

$$P(A | B) = \frac{P(B | A) P(A)}{P(B)}$$

- $P(A | B)$ : Probabilité de $A$ sachant $B$
- $P(B | A)$: Probabilité de $B$ sachant $A$
- $P(A)$ : Probabilité de $A$
- $P(B)$ : Probabilité de $B$

---

## 8. Variables Aléatoires et Distributions

### Espérance de la variable aléatoire

$$E(X) = \sum x_i P(x_i)$$

- $E(X)$ : Espérance de la variable aléatoire $X$
- $x_i$ : Valeurs possibles de $X$
- $P(x_i)$ : Probabilité associée à $x_i$

---

### Variance d’une variable aléatoire

$$Var(X) = E(X^2) - (E(X))^2 $$

- $Var(X)$ : Variance de $X$
- $E(X^2)$ : Espérance de $X^2$
- $E(X)$ : Espérance de $X$

---

## 9. Inférence Statistique

### Intervalle de confiance pour une moyenne

$$IC = \bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

- $IC$ : Intervalle de confiance pour la moyenne
- $\bar{x}$ : Moyenne de l’échantillon
- $z_{\alpha/2}$ : Valeur critique de la distribution normale
- $\sigma$: Écart-type de l’échantillon
- $n$ : Taille de l’échantillon

---

### Test d’hypothèse

- $H_0$ : Hypothèse nulle
- $H_1$ : Hypothèse alternative

Utilise une statistique de test (calculée à partir des données) pour comparer avec un seuil critique afin de décider si $H_0$ est rejetée.

---

### Test t de Student

$$t = \frac{\bar{x} - \mu}{\frac{s}{\sqrt{n}}}$$

- $t$ : Statistique de test
- $\bar{x}$ : Moyenne de l’échantillon
- $\mu$ : Moyenne hypothétique de la population
- $s$ : Écart-type de l’échantillon
- $n$ : Taille de l’échantillon

---

### Test du Chi-2

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

- $\chi^2$ : Statistique de test du chi-deux
- $O_i$ : Observations (fréquences observées)
- $E_i$ : Fréquences attendues

---
