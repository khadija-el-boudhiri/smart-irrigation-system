import os
import pandas as pd

df = pd.read_csv("data/raw/data.csv")

# 1. Nettoyer altitude (enlever le tiret)
df['altitude'] = pd.to_numeric(
    df['altitude'].astype(str).str.replace('-', '', regex=False),
    errors='coerce'
)
df = df.dropna(subset=['altitude'])

# 2. Supprimer lignes corrompues
df = df[df['temperature'] < 100]
df = df[df['pressure'] > 0]

# 3. Calibrer soil moisture → pourcentage (0-100%)
df['soil_pct'] = ((df['soilmiosture'] - df['soilmiosture'].min()) /
                  (df['soilmiosture'].max() - df['soilmiosture'].min()) * 100).round(2)

# 4. Combiner date + time en timestamp
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True)

# 5. Supprimer colonnes inutiles
df = df.drop(columns=['id', 'date', 'time', 'soilmiosture'])

# 6. Sauvegarder
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/features.csv", index=False)
print(f"Dataset nettoyé : {len(df)} lignes")