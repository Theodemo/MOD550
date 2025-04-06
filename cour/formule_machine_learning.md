# Formules Mathématiques en Machine Learning

## 1. Algèbre linéaire

### Vecteurs, matrices, tenseurs

- **Vecteur** : Un vecteur est une liste ordonnée de nombres. En Machine Learning, il est souvent utilisé pour représenter des données d’entrée (features). 
  - Forme :  
    $$
    \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
    $$
  - Où $v_1, v_2, \dots, v_n$ sont les éléments du vecteur et $n$ est la dimension.

- **Matrice** : Une matrice est un tableau de nombres organisé en lignes et colonnes. Elle est utilisée pour représenter des ensembles de données ou des transformations linéaires.
  - Forme :  
    $$
    \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}
    $$
  - Où $m$ est le nombre de lignes et $n$ est le nombre de colonnes.

- **Tenseur** : Un tenseur est une généralisation des matrices et des vecteurs aux dimensions supérieures (par exemple, une image en 3D est un tenseur).

---

### Produit scalaire et produit matriciel

- **Produit scalaire** (ou produit point) de deux vecteurs $\mathbf{u}$ et $\mathbf{v}$ :
  $$
  \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i
  $$
  - $\mathbf{u}$ et $\mathbf{v}$ sont deux vecteurs de dimension $n$, et $u_i$ et $v_i$ sont les composants de ces vecteurs.
  
- **Produit matriciel** : Le produit de deux matrices $\mathbf{A}$ et $\mathbf{B}$ (avec $\mathbf{A} \in \mathbb{R}^{m \times n}$ et $\mathbf{B} \in \mathbb{R}^{n \times p}$) donne une matrice $\mathbf{C} \in \mathbb{R}^{m \times p}$ :
  $$
  \mathbf{C} = \mathbf{A} \mathbf{B}, \quad c_{ij} = \sum_{k=1}^n a_{ik} b_{kj}
  $$
  - $c_{ij}$ est l'élément de la matrice résultante, obtenu par la somme des produits des éléments correspondants des lignes de $\mathbf{A}$ et des colonnes de $\mathbf{B}$.

---

### Transposition, inversion, trace et déterminant

- **Transposition** : La transposition d'une matrice $\mathbf{A}$ consiste à échanger ses lignes et ses colonnes.  
  $$
  \mathbf{A}^T = \begin{bmatrix} a_{11} & a_{21} & \dots \\ a_{12} & a_{22} & \dots \\ \vdots & \vdots & \ddots \end{bmatrix}
  $$

- **Inversion** : L'inverse d'une matrice $\mathbf{A}$, notée $\mathbf{A}^{-1}$, est la matrice qui, multipliée par $\mathbf{A}$, donne la matrice identité $\mathbf{I}$.  
  $$
  \mathbf{A} \mathbf{A}^{-1} = \mathbf{I}
  $$
  L'inverse n'existe que si $\mathbf{A}$ est carrée et non singulière (son déterminant est non nul).

- **Trace** : La trace d'une matrice carrée $\mathbf{A}$ est la somme de ses éléments diagonaux.  
  $$
  \text{Tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}
  $$

- **Déterminant** : Le déterminant d'une matrice $\mathbf{A}$ est une valeur scalaire qui permet de déterminer si la matrice est inversible.  
  $$
  \text{det}(\mathbf{A})
  $$

---

### Décomposition en valeurs propres et SVD

- **Valeurs propres** : Pour une matrice carrée $\mathbf{A}$, une valeur propre $\lambda$ est une constante qui satisfait $\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$, où $\mathbf{v}$ est un vecteur propre.  
  $$
  \text{det}(\mathbf{A} - \lambda \mathbf{I}) = 0
  $$

- **Décomposition en valeurs singulières (SVD)** : La décomposition SVD d'une matrice $\mathbf{A}$ est donnée par $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$, où :
  - $\mathbf{U}$ et $\mathbf{V}$ sont des matrices orthogonales,
  - $\mathbf{\Sigma}$ est une matrice diagonale contenant les valeurs singulières.

---

### Normes de vecteurs : L1, L2, Frobenius

- **Norme L1** :  
  $$
  \| \mathbf{v} \|_1 = \sum_{i=1}^n |v_i|
  $$

- **Norme L2** (Euclidienne) :  
  $$
  \| \mathbf{v} \|_2 = \sqrt{\sum_{i=1}^n v_i^2}
  $$

- **Norme de Frobenius** (pour les matrices) :  
  $$
  \| \mathbf{A} \|_F = \sqrt{\sum_{i,j} |a_{ij}|^2}
  $$

---

### Projection et changement de base

- **Projection** : La projection d'un vecteur $\mathbf{v}$ sur un vecteur $\mathbf{u}$ est donnée par :  
  $$
  \text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u}
  $$

- **Changement de base** : Si $\mathbf{v}$ est un vecteur dans un espace vectoriel et $P$ est une matrice de changement de base, alors :  
  $$
  \mathbf{v}' = P^{-1} \mathbf{v}
  $$

---

## 2. Calcul différentiel

### Dérivées partielles

- **Dérivée partielle** : La dérivée partielle de $f(x_1, x_2, \dots, x_n)$ par rapport à $x_i$ est définie par :  
  $$
  \frac{\partial f}{\partial x_i} = \lim_{\Delta x_i \to 0} \frac{f(x_1, \dots, x_i + \Delta x_i, \dots, x_n) - f(x_1, \dots, x_n)}{\Delta x_i}
  $$

---

### Gradient, jacobien et hessien

- **Gradient** : Le gradient de $f(x)$ est un vecteur contenant toutes les dérivées partielles de $f$ par rapport à ses variables.  
  $$
  \nabla f(x) = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)
  $$

- **Jacobian** : Le jacobien est la matrice des dérivées partielles de chaque fonction composant un vecteur-valued function $\mathbf{f}$.  
  $$
  J_{\mathbf{f}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}
  $$

- **Hessien** : La matrice hessienne est la matrice des dérivées secondes d'une fonction scalaire.  
  $$
  H(f) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}
  $$

---

### Règle de la chaîne (backpropagation)

- **Règle de la chaîne** : La dérivée d'une fonction composée est donnée par :  
  $$
  \frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)
  $$
  En backpropagation, cela est utilisé pour propager les erreurs à travers un réseau neuronal et ajuster les poids.

---

### Optimisation : descente de gradient (batch, stochastique, mini-batch)

- **Descente de gradient** : L'algorithme de base pour l'optimisation consiste à mettre à jour les paramètres $\theta$ selon :  
  $$
  \theta := \theta - \eta \nabla_\theta J(\theta)
  $$
  Où $J(\theta)$ est la fonction de coût et $\eta$ est le taux d'apprentissage.

- **Descente de gradient stochastique (SGD)** : La mise à jour est effectuée sur un seul échantillon à chaque itération.  
- **Mini-batch gradient descent** : Un compromis entre la descente de gradient stochastique et par batch, où la mise à jour est effectuée après avoir traité un sous-ensemble de données (mini-batch).

---

## 3. Probabilités et statistiques

### Espérance mathématique, variance, covariance

- **Espérance** : L'espérance (ou moyenne) d'une variable aléatoire $X$ est :  
  $$
  \mathbb{E}[X] = \sum_{i=1}^n p(x_i) x_i
  $$

- **Variance** : La variance mesure la dispersion des valeurs autour de la moyenne :  
  $$
  \text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
  $$

- **Covariance** : La covariance mesure la relation linéaire entre deux variables aléatoires $X$ et $Y$ :  
  $$
  \text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
  $$

---

### Loi normale, loi de Bernoulli, loi binomiale, loi de Poisson

- **Loi normale** :  
  $$
  f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
  $$
  Où $\mu$ est la moyenne et $\sigma^2$ est la variance.

- **Loi de Bernoulli** :  
  $$
  P(X=1) = p, \quad P(X=0) = 1-p
  $$

- **Loi binomiale** :  
  $$
  P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
  $$

- **Loi de Poisson** :  
  $$
  P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
  $$
  Où $\lambda$ est le taux moyen d'occurrences.

---

### Théorème de Bayes

- **Théorème de Bayes** :  
  $$
  P(A|B) = \frac{P(B|A) P(A)}{P(B)}
  $$

---

### Entropie, entropie croisée

- **Entropie de Shannon** :  
  $$
  H(X) = -\sum_{i=1}^n p(x_i) \log p(x_i)
  $$

- **Entropie croisée** :  
  $$
  H(p, q) = -\sum_{i=1}^n p(x_i) \log q(x_i)
  $$

---

### Maximum de vraisemblance (MLE) et maximum a posteriori (MAP)

- **Maximum de vraisemblance** :  
  $$
  \hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \prod_{i=1}^n p(x_i|\theta)
  $$

- **Maximum a posteriori** :  
  $$
  \hat{\theta}_{MAP} = \arg\max_\theta p(\theta | X) = \arg\max_\theta \frac{p(X|\theta) p(\theta)}{p(X)}
  $$

---

### Indépendance, corrélation, loi conjointe

- **Indépendance** : Deux variables aléatoires $X$ et $Y$ sont indépendantes si $P(X, Y) = P(X)P(Y)$.

- **Corrélation** : La corrélation entre $X$ et $Y$ est donnée par :  
  $$
  \rho(X, Y) = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
  $$

- **Loi conjointe** : La probabilité conjointe de $X$ et $Y$ est :  
  $$
  P(X, Y) = P(X|Y) P(Y)
  $$

---

## 4. Apprentissage supervisé
### Fonction de coût : MSE, MAE, log-loss  
### Régression linéaire : équation normale, fonction de coût  
### Régression logistique : sigmoid, log-loss  
### SVM : fonction de marge, dualité, noyaux (kernel trick)  
### Arbres de décision : gain d'information, entropie  

## 5. Apprentissage non supervisé
### K-means : distance euclidienne, inertie  
### PCA : variance expliquée, décomposition en vecteurs propres  
### Clustering hiérarchique : linkage, dendrogramme  
### Réduction de dimension : t-SNE, UMAP (concepts + distances)  

## 6. Réseaux de neurones et deep learning
### Fonctions d'activation : sigmoid, tanh, ReLU, softmax  
### Forward pass et backward pass  
### Fonction de perte : entropie croisée, MSE  
### Rétropropagation : règles de dérivation  
### CNN : convolution, padding, stride  
### RNN, LSTM, GRU : mémoire, portes, état caché  

## 7. Mesures de performance
### Accuracy, précision, rappel, F1-score  
### Matrice de confusion  
### Courbes ROC et AUC  
### Courbe precision-recall  
### Score de silhouette (clustering)  

## 8. Régularisation et généralisation
### L1 (Lasso), L2 (Ridge)  
### Dropout  
### Early stopping  
### Biais-variance (erreur quadratique totale = biais² + variance + bruit)  

## 9. Méthodes bayésiennes
### Théorème de Bayes appliqué aux modèles  
### Inférence bayésienne  
### Gaussian Processes (GP)  
### Approximations : Monte Carlo, variational inference  

## 10. Approches Avancées

### 10.1 Information Theory (Théorie de l'Information)
### Divergence de Kullback-Leibler (KL Divergence)  
### Information mutuelle  
### Entropie de Shannon  

### 10.2 Optimization Avancée
### Méthodes de Newton  
### Méthodes du Lagrangien et multiplicateurs de Lagrange  
### Conditions KKT (Karush-Kuhn-Tucker)  
### Optimisation sous contraintes  

### 10.3 Apprentissage par Renforcement (RL)
### Espérance de récompense  
### Équation de Bellman  
### Q-learning  
### Policy Gradients  

### 10.4 Théorie de la Généralisation
### Dimension VC (Vapnik-Chervonenkis)  
### Bornes de l'apprentissage PAC (Probably Approximately Correct)  
### Complexité de Rademacher  

### 10.5 Graphes et Graph Neural Networks (GNN)
### Matrices d'adjacence  
### Laplacien de graphe  
### Convolution sur graphe  

### 10.6 Apprentissage Semi-Supervisé & Auto-Supervisé
### Contraste de similarité  
### Perte de type InfoNCE (Noise Contrastive Estimation)  

## 11. Applications Spécifiques

### 11.1 Traitement du Langage Naturel (NLP)
### Mécanisme d'attention  
### Self-attention et Transformer  
### Embeddings (Word2Vec, GloVe)  
### Codage positionnel (Positional Encoding)  

### 11.2 Vision par Ordinateur
### Augmentation de données  
### Perte contrastive  
### Intersection over Union (IoU)  

### 11.3 Séries Temporelles
### Autocorrélation  
### Stationnarité  
### Décomposition des séries temporelles (tendance, saisonnalité, bruit)  

---
