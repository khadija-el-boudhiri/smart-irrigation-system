import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/processed/features.csv")

# 1. Encoder la colonne 'class' (texte → chiffre)
# Very Dry=0, Dry=1, Wet=2, Very Wet=3
class_map = {'Very Dry': 0, 'Dry': 1, 'Wet': 2, 'Very Wet': 3}
df['class_encoded'] = df['class'].map(class_map)

# 2. Extraire des features utiles depuis timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day']  = df['timestamp'].dt.day

# 3. Supprimer colonnes inutiles pour le modèle
df = df.drop(columns=['class', 'timestamp'])

# 4. Vérifier
print(df.dtypes)
print(df.head(3))
print(f"\n✅ Shape final : {df.shape}")
print(f"Distribution target 'status' :\n{df['status'].value_counts()}")

# 5. Sauvegarder
df.to_csv("data/processed/features_ready.csv", index=False)
print("\n✅ features_ready.csv sauvegardé !")