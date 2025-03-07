import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier  # importerar EBM

# Läs in datasetet
df = pd.read_json('UCI Heart Disease Prediction.json')

# Separera features (X) och målvariabel (y)
X = df.drop(columns=['target'])
y = df['target']

# Skapa och träna modellen
model = ExplainableBoostingClassifier()  # Skapar en EBM-modell
model.fit(X, y)  # Tränar modellen med data (X = funktioner, y = "target")

# Hämta feature-importance (global explanation)
features = model.term_names_  # Hämtar namnen på funktioner
feature_importances = model.term_importances()  # Hämtar importance-värden för varje funktion (abs!)

# Visualisera global feature-importance (figur 1)
plt.figure(figsize=(10, 5))  # Skapar en ny figur
plt.barh(features, feature_importances, color='skyblue')  # (funktion, funktions-värde)
plt.xlabel("Viktighetsvärde")
plt.ylabel("Funktion")
plt.title("Funktioners betydelse - Global Förklaring (EBM)")
plt.gca().invert_yaxis()  # Inverterar Y-axeln så den viktigaste funktionen kommer först

# Lokal förklaring för rad 200 (figur 2)
if 200 < len(X):
    instance = X.iloc[[200]]
    prediction = model.predict(instance)[0]
    explanation = model.explain_local(instance)

    print(f"Lokal förklaring för rad 200: Förutspådd etikett = {prediction}")
    print("Funktioners bidrag till prediktionen:")

    # Skriv ut de specifika funktionernas bidrag i terminalen
    for feature, importance in zip(features, explanation.data(0)["scores"]):
        print(f"{feature}: {importance:.4f}")
    
    # Visualisera den lokala förklaringen (figur 2) som en barplot
    local_importances = explanation.data(0)["scores"]
    plt.figure(figsize=(10, 5))  # Skapar en ny figur
    plt.barh(features, local_importances, color='lightgreen')
    plt.xlabel("Bidrag till prediktion")
    plt.ylabel("Funktion")
    plt.title("Lokal Förklaring för Rad 200")
    plt.gca().invert_yaxis()  # Inverterar Y-axeln för att visa den mest bidragande först

else:
    print("Rad 200 finns inte i datasetet.")

# Visa båda figurerna samtidigt
plt.show()
