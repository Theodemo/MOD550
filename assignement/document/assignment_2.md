## Deadline: 18.02

### **Tâches à réaliser**

1. **Correction du code `mse_scaling_2.py`**  
   - Assurez-vous que `mse_vanilla == mse_numpy == mse_ske`.
   - Affichez le temps d'exécution de chaque approche.

2. **Création de données**  
   - Générer une fonction oscillatoire 1D avec et sans bruit.

3. **Clustering des données**  
   - Choisir une méthode de clustering.
   - Grouper les données et tracer la variance en fonction du nombre de clusters.

4. **Régression des données**  
   - Appliquer **LR (Linear Regression)**, **NN (Neural Networks)** et **PINNs (Physics-Informed Neural Networks)**.

5. **Visualisation des solutions**  
   - Tracer la solution en fonction du nombre d'itérations exécutées (NN et PINNs).

6. **Évaluation de l'erreur**  
   - Tracer l'erreur par rapport à la vérité en fonction des itérations (LR, NN et PINNs).

7. **Monitoring du progrès**  
   - Supposer que la vérité est inconnue.
   - Sélectionner une méthode pour surveiller la progression des calculs et tracer son évolution (LR, NN et PINNs).

8. **Exécution du script de reinforcement learning**

---

### **Livrable**

#### **Format**
- Dépôt sur **GitHub**.
- Assurez-vous que l'accès à votre dépôt m'est accordé.
- Utilisation du dossier `MOD550/code`.
- Sauvegarde du programme en **.py** (si `ipynb`, convertir en `.py`).
- Nom du programme : **`exercise_2.py`**.

#### **Exécution du programme (`python exercise_2.py`)**
1. Afficher `'Test successful'` si la tâche 1 est réussie.
2. Afficher `f'Data generated: {n_points}, {range}, {other_info}'` (demandez des précisions si nécessaire).
3. Afficher les informations sur la méthode de clustering et ses paramètres.
   - Tracer la variance en fonction du nombre de clusters.
4. Afficher `'Task completed {regression_method}'` après chaque régression.
5. Tracer la fonction de régression en fonction du nombre d'itérations exécutées.
6. Tracer l'erreur en fonction du nombre d'itérations.
7. Tracer la variable de suivi du progrès.
   - **Bonus** : Superposer les graphes des points 5, 6 et 7.
8. Tracer le **nombre d'itérations nécessaires pour converger** en fonction du **learning rate**.

---

**Remarque :** Votre programme doit être téléchargeable et exécutable en tant que script. Soyez précis dans la **nomination des fichiers**.

