# **Norme pondérée de l'erreur**

La norme pondérée entre les prédictions $\hat{Y}$ et les observations $Y$ est donnée par :


$$| Y - \hat{Y} \|_W = \sqrt{(Y - \hat{Y})^T W (Y - \hat{Y})}$$


Avec $W$ défini comme :


$$ W = \text{diag} \left( \frac{1}{\theta_1^2}, \frac{1}{\theta_2^2}, \dots, \frac{1}{\theta_n^2} \right) $$


Cela peut aussi s'écrire sous forme développée :


 
$$| Y - \hat{Y} |_W = \sqrt{\sum_{i=1}^{n} \frac{(y_i - \hat{y}_i)^2}{\theta_i^2}} $$


## Explication des paramètres

- $Y$  : Vecteur des observations ($n \times 1$).
- $\hat{Y}$ : Vecteur des prédictions ($n \times 1$).
- $W$ : Matrice diagonale contenant les poids $W = \text{diag}(1/\theta_1^2, 1/\theta_2^2, \dots, 1/\theta_n^2)$.
- $\theta_i$ : Écart-type ou incertitude associée à l'observation $y_i$.

Cette norme accorde une pondération plus faible aux erreurs associées à des observations ayant une grande incertitude $\theta_i$.

# **Filtre de Kalman d'Ensemble (EnKF)**

L'Ensemble Kalman Filter (EnKF) est une méthode d'assimilation de données utilisée pour estimer l'état d'un système dynamique en combinant des observations et un modèle d'évolution.

### **1. Prévision de l'ensemble :**
Chaque membre de l'ensemble évolue selon le modèle dynamique :


$$x_k^{(i)} = M(x_{k-1}^{(i)}) + \eta_k^{(i)}$$

où :
- $x_k^{(i)}$ est le $i$-ème membre de l'ensemble des états à l'instant $k$.
- $M(\cdot)$ est le modèle d'évolution de l'état.
- $\eta_k^{(i)}$ est un bruit du modèle supposé gaussien.

### **2. Perturbation des observations :**
Les observations sont perturbées pour chaque membre :


$$y_k^{(i)} = y_k + \epsilon_k^{(i)}$$

où :
- $y_k$ est le vecteur des observations à l'instant $k$.
- $\epsilon_k^{(i)} \sim \mathcal{N}(0, R)$ est un bruit gaussien avec covariance $R$.

### **3. Mise à jour de l'ensemble :**
Chaque membre de l’ensemble est corrigé en fonction des observations :


$$x_k^{(i)} = x_k^{(i)} + K_k \left( y_k^{(i)} - H x_k^{(i)} \right)$$

où :
- $H$ est la matrice d'observation qui projette l'état dans l'espace des observations.
- $K_k$ est le **gain de Kalman**, défini par :


$$K_k = P_k H^T (H P_k H^T + R)^{-1}$$

- $P_k$ est la matrice de covariance de l’ensemble, approximée par :


$$P_k \approx \frac{1}{N - 1} \sum_{i=1}^{N} (x_k^{(i)} - \bar{x}_k)(x_k^{(i)} - \bar{x}_k)^T$$

avec $\bar{x}_k$ la moyenne de l’ensemble :


$$\bar{x}_k = \frac{1}{N} \sum_{i=1}^{N} x_k^{(i)}$$

### **Explication des paramètres :**
- $x_k^{(i)}$ : $i$-ème membre de l’ensemble des états à l’instant $k$.
- $M(\cdot)$ : Modèle d’évolution de l’état.
- $\eta_k^{(i)}$ : Bruit du modèle.
- $y_k$ : Vecteur des observations à l’instant $k$.
- $\epsilon_k^{(i)}$ : Bruit ajouté aux observations.
- $H$ : Matrice d'observation.
- $R$ : Matrice de covariance des observations.
- $K_k$ : Gain de Kalman utilisé pour la correction.
- $P_k$ : Matrice de covariance de l’ensemble.
- $\bar{x}_k$ : Moyenne des états de l'ensemble.
- $N$ : Nombre de membres de l’ensemble.

L'EnKF est largement utilisé en météorologie, océanographie et d'autres domaines nécessitant l'assimilation de données en grande dimension.

# **L'Ensemble Smoother avec Multiple Data Assimilation (ESMDA)**

L'Ensemble Smoother avec Multiple Data Assimilation (ESMDA) est une méthode d'assimilation de données basée sur les ensembles. Contrairement à l'EnKF, qui assimile les observations séquentiellement, l'ESMDA assimile les observations en plusieurs étapes en ajustant progressivement l'ensemble des états.

### **1. Initialisation de l'ensemble :**
On considère un ensemble de $N$ états initialement donné :


$$x_0^{(i)}, \quad i = 1, \dots, N$$

où $x_0^{(i)}$ représente les membres de l'ensemble initial.

### **2. Assimilation des observations en plusieurs étapes :**
L'ESMDA divise l'assimilation en $m$ étapes. À chaque étape $j$, les observations $y$ sont perturbées par un bruit gaussien :


$$y^{(i, j)} = y + \epsilon^{(i, j)}, \quad \epsilon^{(i, j)} \sim \mathcal{N}(0, \alpha_j R)$$

où $\alpha_j$ est un facteur de pondération défini par :


$$\alpha_j = \frac{m}{\text{nombre d'étapes}}$$

À chaque étape, la mise à jour des membres de l'ensemble suit la correction :


$$x_j^{(i)} = x_{j-1}^{(i)} + K_j \left( y^{(i, j)} - H x_{j-1}^{(i)} \right)$$

où $K_j$ est le **gain de Kalman généralisé** :


$$K_j = P_j H^T (H P_j H^T + \alpha_j R)^{-1}$$

La matrice de covariance de l'ensemble $P_j$ est estimée par :


$$P_j \approx \frac{1}{N - 1} \sum_{i=1}^{N} (x_j^{(i)} - \bar{x}_j)(x_j^{(i)} - \bar{x}_j)^T$$

avec $\bar{x}_j$ la moyenne des états de l'ensemble à l'étape $j$ :


$$\bar{x}_j = \frac{1}{N} \sum_{i=1}^{N} x_j^{(i)}$$

### **3. Mise à jour finale :**
Après $m$ étapes, la solution finale est donnée par l'ensemble mis à jour $x_m^{(i)}$.

---

### **Explication des paramètres :**

- $x_j^{(i)}$ : $i$-ème membre de l’ensemble des états à l’étape $j$.
- $y$ : Vecteur des observations.
- $\epsilon^{(i, j)}$ : Bruit ajouté aux observations à l’étape $j$.
- $H$ : Matrice d'observation.
- $R$ : Matrice de covariance des observations.
- $m$ : Nombre d'étapes d'assimilation.
- $\alpha_j$: Facteur de pondération $\alpha_j = \frac{m}{\text{nombre d'étapes}}$.
- $K_j$ : Gain de Kalman généralisé pour l'étape $j$.
- $P_j$ : Matrice de covariance de l'ensemble à l'étape $j$.
- $\bar{x}_j$ : Moyenne des états de l'ensemble à l'étape $j$.
- $N$ : Nombre de membres de l’ensemble.

L'ESMDA est particulièrement utilisé pour les problèmes d'inversion en ingénierie pétrolière et en hydrologie, où plusieurs mises à jour successives permettent une meilleure assimilation des observations.

# **Iterative Ensemble Smoother (IES)**

L'Iterative Ensemble Smoother (IES) est une méthode d'assimilation de données pour les systèmes dynamiques non linéaires, où la mise à jour des états se fait de manière itérative. L’IES utilise un ensemble d'états et procède à une série d'itérations pour affiner les estimations des états en fonction des observations.

### **1. Initialisation de l'ensemble :**

On commence par un ensemble de $N$ états, noté $x_0^{(i)}$, pour $i = 1, \dots, N$. Ces états sont initialement générés à partir d'une distribution (généralement gaussienne).


$$x_0^{(i)}, \quad i = 1, \dots, N$$

### **2. Iterative Ensemble Smoothing (Mise à jour itérative) :**
L'IES procède par itérations pour affiner la mise à jour des membres de l'ensemble. À chaque itération $k$, l’état $x_{k-1}^{(i)}$ est mis à jour en fonction des observations $y_k$ :


$$x_k^{(i)} = x_{k-1}^{(i)} + K_k \left( y_k^{(i)} - H x_{k-1}^{(i)} \right)$$

où :
- $K_k$ est le **gain de Kalman** à l’itération $k$, donné par :


$$K_k = P_k H^T (H P_k H^T + R)^{-1}$$

et où $P_k$ est la matrice de covariance de l’ensemble estimée par :


$$P_k \approx \frac{1}{N - 1} \sum_{i=1}^{N} (x_{k-1}^{(i)} - \bar{x}_{k-1})(x_{k-1}^{(i)} - \bar{x}_{k-1})^T$$

avec $\bar{x}_{k-1}$ la moyenne des états à l’itération précédente :


$$\bar{x}_{k-1} = \frac{1}{N} \sum_{i=1}^{N} x_{k-1}^{(i)}$$

### **3. Répétition de l'itération :**
L'itération est répétée jusqu'à ce que la mise à jour des états converge, c’est-à-dire que les ajustements entre les itérations deviennent suffisamment petits, ou après un nombre défini d'itérations.

### **4. Mise à jour finale :**
Après les $k$ itérations, les membres de l'ensemble sont mis à jour pour donner la solution finale $x_k^{(i)}$ pour $i = 1, \dots, N$.

---

### **Explication des paramètres :**

- $x_k^{(i)}$ : $i$-ème membre de l’ensemble des états après l’itération $k$.
- $y_k$ : Vecteur des observations à l’itération $k$.
- $H$ : Matrice d'observation qui projette l'état dans l’espace des observations.
- $R$ : Matrice de covariance des observations.
- $K_k$ : Gain de Kalman à l'itération $k$.
- $P_k$ : Matrice de covariance de l’ensemble à l’itération $k$.
- $\bar{x}_{k-1}$ : Moyenne des états de l'ensemble à l’itération $k-1$.
- $N$ : Nombre de membres de l’ensemble.
- $k$ : Numéro de l’itération actuelle.

L'IES est couramment utilisé pour des problèmes d'inversion en géophysique, modélisation climatique, ou encore en météorologie, où la mise à jour itérative améliore la précision des estimations de l'état.



# **Algorithmes d'optimisation sans gradient**

Les **méthodes sans gradient** sont des algorithmes d'optimisation qui ne nécessitent pas le calcul explicite du gradient de la fonction de coût. Elles sont particulièrement utiles pour les problèmes où le calcul du gradient est difficile, coûteux ou impossible (par exemple, dans les problèmes non différentiables ou les modèles très complexes).

#### **1. Formulation du problème :**
On cherche à minimiser une fonction de coût $J(x)$, où  $x$ est un vecteur d'état ou de paramètres à optimiser. La méthode sans gradient cherche $x^*$ qui minimise $J(x)$ sans utiliser de dérivées.

$$x^* = \arg\min_x J(x)$$

## **1. Recherche par grille (Grid Search)**
La **Recherche par grille** consiste à évaluer la fonction de coût sur une grille de valeurs de \( x \). L'algorithme explore toutes les combinaisons possibles dans une plage prédéfinie pour chaque paramètre.

### **Formule**:

$$x^* = \arg\min_x J(x), \quad x \in \text{Grille}$$

### **Paramètres**:
- $x$ : Paramètres ou vecteur d'état de l'algorithme.
- $J(x)$ : Fonction de coût à minimiser.
- **Grille** : Un ensemble discret de points dans l'espace de recherche.

---

## **2. Recherche aléatoire (Random Search)**
La **Recherche aléatoire** génère des valeurs aléatoires de $x$ et évalue la fonction de coût à chaque point pour trouver la solution optimale.

### **Formule**:

$$x^* = \arg\min_x J(x), \quad x \sim \mathcal{U}(\text{borne inférieure}, \text{borne supérieure})$$

### **Paramètres**:
- $x$ : Paramètre ou vecteur d'état.
- $J(x)$ : Fonction de coût.
- $\mathcal{U}$ : Distribution uniforme dans l'espace de recherche, entre la borne inférieure et supérieure.

---

## **3. Algorithmes évolutionnaires (Evolutionary Algorithms)**
Les **algorithmes évolutionnaires** incluent des techniques comme les **algorithmes génétiques** et **Differential Evolution**. Ces méthodes utilisent des populations de solutions et des opérateurs génétiques pour évoluer vers une solution optimale.

### **Formule (Algorithmes génétiques)**:
1. Sélectionner une population $P_k$ de solutions à l'itération $k.
2. Appliquer des opérateurs génétiques (croisement, mutation) pour produire la population $P_{k+1}$ à l’itération suivante.


$$P_{k+1} = \text{Croisement}(P_k) \cup \text{Mutation}(P_k)$$

### **Paramètres**:
- $P_k$ : Population de solutions à l’itération $k$.
- **Croisement** : Opération qui mélange les solutions (parents) pour créer de nouvelles solutions (enfants).
- **Mutation** : Opération qui modifie aléatoirement une solution pour introduire de la diversité.
- **Taux de croisement et mutation** : Probabilités d'appliquer ces opérateurs.

---

## **4. Simulated Annealing (Recuit simulé)**
Le **Simulated Annealing** est inspiré du processus de refroidissement des métaux. Il commence avec une température élevée et la diminue lentement pour permettre à l'algorithme d'explorer l'espace de recherche.

### **Formule**:
$$x_{k+1} = \begin{cases}
x_k & \text{si } J(x_{k+1}) < J(x_k) \\
x_k + \Delta x & \text{si } J(x_{k+1}) \geq J(x_k) \text{ et avec probabilité } P(T)
\end{cases}$$

$$P(T) = \exp\left( -\frac{J(x_{k+1}) - J(x_k)}{T} \right)$$

### **Paramètres**:
- $x_k$ : Solution à l'itération $k$.
- $J(x_k)$ : Fonction de coût à l’itération $k$.
- $\Delta x$ : Perturbation aléatoire appliquée à la solution.
- $T$ : Température, qui diminue à chaque itération.
- $P(T)$ : Probabilité d’accepter une solution moins bonne en fonction de la température.

---

## **5. Particle Swarm Optimization (PSO)**
**Particle Swarm Optimization (PSO)** est un algorithme qui simule un essaim de particules. Chaque particule se déplace dans l'espace de recherche en fonction de sa propre expérience et de l'expérience collective de l'essaim.

### **Formule**:
1. Mettre à jour la vitesse de chaque particule :

$$v_i^{(t+1)} = w v_i^{(t)} + c_1 r_1 (p_i - x_i^{(t)}) + c_2 r_2 (g - x_i^{(t)})$$
2. Mettre à jour la position de chaque particule :

$$x_i^{(t+1)} = x_i^{(t)} + v_i^{(t+1)}$$

### **Paramètres**:
- $v_i^{(t)}$ : Vitesse de la particule $i$ à l'itération $t$.
- $x_i^{(t)}$ : Position de la particule $i$ à l'itération $t$.
- $p_i$ : Meilleur emplacement de la particule $i$.
- $g$ : Meilleur emplacement global de l’essaim.
- $c_1, c_2$ : Coefficients d'accélération, contrôlant l'influence de l'expérience personnelle et collective.
- $r_1, r_2$ : Variables aléatoires uniformes sur [0,1].
- $w$ : Facteur de relaxation de la vitesse.
  
---


# **Méthodes basées sur le gradient (Gradient-Based Methods)**

Les **méthodes basées sur le gradient** utilisent les informations dérivées de la fonction de coût pour guider l'optimisation. Ces méthodes sont efficaces lorsque la fonction de coût est différentiable et que le gradient peut être facilement calculé.

#### **1.Formulation du problème :**
On cherche à minimiser la fonction de coût $J(x)$ par rapport à $x$, où $x$ est un vecteur de paramètres :


$$x^* = \arg\min_x J(x)$$

Les **méthodes basées sur le gradient** cherchent le minimum de $J(x)$ en suivant le sens du gradient (la pente la plus raide) de $J(x)$.


## 1. Descente de Gradient (Gradient Descent)

**Formule** :

$$x^{(k+1)} = x^{(k)} - \alpha \nabla J(x^{(k)})$$

**Paramètres** :
- $x^{(k)}$ : vecteur de paramètres à l'itération $k$
- $\alpha$ : taux d'apprentissage (**learning rate**), un petit nombre positif
- $\nabla J(x^{(k)})$: gradient de la fonction de coût $J(x)$ au point $x^{(k)}$

> Utilisé pour minimiser une fonction $J(x)$ en suivant la direction opposée du gradient.

---

## 2. Descente de Gradient Stochastique (Stochastic Gradient Descent - SGD)

**Formule** :

$$x^{(k+1)} = x^{(k)} - \alpha \nabla J(x^{(k)}, \xi)$$

**Paramètres** :
- $\xi$ : un **échantillon** ou un **mini-lot (mini-batch)** tiré aléatoirement de l'ensemble de données
- Les autres paramètres sont les mêmes que pour la descente de gradient classique

> Plus efficace pour les grands ensembles de données. Introduit un certain bruit qui peut aider à éviter les minima locaux.

---

## 3. Méthode de Newton

**Formule** :

$$x^{(k+1)} = x^{(k)} - \alpha H^{-1} \nabla J(x^{(k)})$$

**Paramètres** :
- $H$ : matrice Hessienne de $J(x)$ (matrice des dérivées secondes)
- $H^{-1} \nabla J(x^{(k)})$ : direction de descente ajustée selon la courbure
- $\alpha$ : taux d’apprentissage (parfois pris égal à 1)

> Convergence rapide mais coûteuse à cause du calcul de la Hessienne.

---

## 4. Méthodes de Quasi-Newton (ex: BFGS)

**Formule générale** :

$$x^{(k+1)} = x^{(k)} - \alpha B_k^{-1} \nabla J(x^{(k)})$$

**Paramètres** :
- $B_k$ : approximation de la matrice Hessienne à l’itération $k$
- $B_k^{-1} \nabla J(x^{(k)})$ : direction de mise à jour
- $\alpha$ : souvent déterminé automatiquement par une recherche linéaire (**line search**)

> Évite le calcul exact de la Hessienne, tout en assurant une bonne convergence.

---

#### **3. Exemple de descente de gradient :**
Le processus de descente de gradient consiste à mettre à jour les paramètres à chaque itération en suivant le gradient de la fonction de coût :


$$x^{(k+1)} = x^{(k)} - \alpha \nabla J(x^{(k)})$$

où :
- $x^{(k)}$ : Paramètres à l'itération $k$,
- $\alpha$ : Taux d'apprentissage (taille du pas),
- $\nabla J(x^{(k)})$ : Gradient de la fonction de coût au point $x^{(k)}$.

---