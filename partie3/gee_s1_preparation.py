import ee
import pandas as pd
import numpy as np
import time
from pathlib import Path

def init_gee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate(auth_mode='notebook')
        ee.Initialize()

def extract_s1_monthly_averages(points_df, year=2025, orbit=None):
    """
    Extrait les moyennes mensuelles Sentinel-1 (VV, VH) pour chaque point.
    Version robuste : gère les absences de données et les orbites.

    orbit : None (toutes les orbites), 'ASCENDING' ou 'DESCENDING' pour stabiliser
    la série (réduit le bruit dû au mélange d'orbites).
    """
    
    # 1. Création de la collection Sentinel-1 (GRD, IW)
    s1_col = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterMetadata('instrumentMode', 'equals', 'IW') \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filterDate(f'{year}-01-01', f'{year}-12-31')
    if orbit in ('ASCENDING', 'DESCENDING'):
        s1_col = s1_col.filter(ee.Filter.eq('orbitProperties_pass', orbit))

    # Test rapide de disponibilité
    total_count = s1_col.size().getInfo()
    print(f"Images S1 trouvées pour {year} : {total_count}")
    
    if total_count == 0:
        print(f"AVERTISSEMENT: Aucune image S1 pour {year}. Tentative avec {year-1}...")
        return extract_s1_monthly_averages(points_df, year-1)

    months = range(1, 13)
    monthly_images = []
    
    # Option avancée (non exportée ici) : compter les images par mois (qualité du composite)
    # via m_col.size() ou une bande 'count' par mois pour pondérer l'entraînement côté Python.

    for m in months:
        # Filtrage mensuel
        m_col = s1_col.filter(ee.Filter.calendarRange(m, m, 'month'))
        
        # Si le mois est vide, on prend la moyenne annuelle comme fallback
        # pour éviter d'avoir des colonnes entièrement vides (NaN)
        m_img = ee.Algorithms.If(
            m_col.size().gt(0),
            m_col.mean(),
            s1_col.mean() # Fallback sur la moyenne annuelle si le mois est vide
        )
        m_img = ee.Image(m_img)
        
        # Calcul du ratio VH/VV (S1_GRD est en dB : VH - VV)
        # Note: On utilise des noms de bandes explicites
        vv = m_img.select('VV').rename(f'vv_{m:02d}')
        vh = m_img.select('VH').rename(f'vh_{m:02d}')
        # Calcul sécurisé du ratio
        ratio = vh.subtract(vv).rename(f'ratio_{m:02d}')
        
        monthly_images.append(ee.Image.cat([vv, vh, ratio]))

    # Image composite à 36 bandes
    all_months_img = ee.Image.cat(monthly_images)

    # Conversion des points
    features_list = [
        ee.Feature(ee.Geometry.Point([row['lon'], row['lat']]), {'id': index})
        for index, row in points_df.iterrows()
    ]
    
    # On traite par chunks pour éviter les Timeouts GEE
    data = []
    chunk_size = 2000 # Réduit pour plus de fiabilité
    
    print(f"Extraction S1 pour {len(features_list)} points par lots de {chunk_size}...")
    
    for i in range(0, len(features_list), chunk_size):
        chunk = features_list[i : i + chunk_size]
        print(f"-> Traitement du lot {i//chunk_size + 1} / {int(np.ceil(len(features_list)/chunk_size))}...")
        
        sampled_fc = all_months_img.reduceRegions(
            collection=ee.FeatureCollection(chunk),
            reducer=ee.Reducer.first(),
            scale=10
        )
        
        try:
            chunk_results = sampled_fc.getInfo()['features']
            for f in chunk_results:
                data.append(f['properties'])
        except Exception as e:
            print(f"Erreur lors du lot {i//chunk_size + 1}: {e}")
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    init_gee()
    
    csv_path = PROJECT_ROOT / "dataset" / "points_agricoles.csv"
    if not csv_path.exists():
        print(f"Erreur: {csv_path} n'existe pas.")
        exit(1)
        
    df_points = pd.read_csv(csv_path)
    if "id" not in df_points.columns:
        df_points = df_points.copy()
        df_points.insert(0, "id", np.arange(len(df_points), dtype=np.int64))

    # On extrait pour 2025 (ou 2024 via fallback automatique si vide)
    s1_data = extract_s1_monthly_averages(df_points, year=2025)
    
    # Vérification finale
    nan_count = s1_data.isna().sum().sum()
    print(f"Extraction terminée. NaNs restants : {nan_count}")
    
    if nan_count > 0:
        print("Remplissage des NaNs restants par 0...")
        s1_data = s1_data.fillna(0)

    out_path = PROJECT_ROOT / "partie3" / "s1_monthly_timeseries.csv"
    s1_data.to_csv(out_path, index=False)
    print(f"Séries temporelles S1 sauvegardées dans {out_path}")
