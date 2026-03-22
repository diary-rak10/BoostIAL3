

---

#  Prédiction du Diabète avec un Réseau de Neurones (TensorFlow)

Ce projet Python permet de prédire si un patient est susceptible d’être diabétique en se basant sur des données médicales. L'intelligence artificielle développée ici repose sur un modèle de **réseau de neurones artificiels (ANN)** entraîné à l'aide de **TensorFlow Keras**.

##  Modèle IA utilisé

Le modèle d’IA utilisé dans ce projet est un **réseau de neurones supervisé** structuré comme suit :

* **Couche d’entrée** avec 8 neurones correspondant aux 8 caractéristiques du patient.
* **Première couche cachée** de 6 neurones avec la fonction d’activation **ReLU**.
* **Deuxième couche cachée** de 4 neurones avec **ReLU** également.
* **Couche de sortie** avec 1 neurone et la fonction d’activation **sigmoid** pour produire une probabilité binaire (diabétique ou non).

Ce modèle est entraîné avec l’optimiseur **Adam** et la fonction de perte **binary\_crossentropy**, pendant **100 époques** avec un **batch size** de 10.

 **Précision du modèle sur les données de test : environ 77,92%**

> Le modèle est sauvegardé sous le nom `model.h5` pour être réutilisé sans réentraînement.

##  Données attendues

Le fichier `diabetes.csv` doit contenir les colonnes suivantes :

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome

##  Fonctionnement du programme

1. **Chargement et vérification** des données (`diabetes.csv`)
2. **Prétraitement** : séparation, normalisation des données
3. **Entraînement ou chargement** automatique du modèle existant
4. **Saisie des informations médicales** d’un patient via la console
5. **Prédiction du risque de diabète** et affichage du pourcentage de probabilité

##  Lancer le programme

Installez les dépendances nécessaires :

```bash
pip install pandas numpy scikit-learn tensorflow
```

Exécutez ensuite le fichier Python :

```bash
python votre_script.py
```

Une interface en ligne de commande vous permettra d’entrer les données d’un patient.

##  Exemple d'affichage

```
 Ce patient est probablement **NON DIABÉTIQUE** (92.14% de probabilité).
```

---


