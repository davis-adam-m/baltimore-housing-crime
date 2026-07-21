# import dependencies
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
import re

# load CSA dataset
csa = pd.read_csv('clean_data/csa.csv')

# convert geometry string column back into geometry objects
csa['geometry'] = csa['geometry'].apply(wkt.loads)

# build GeoDataFrame
csa_gdf = gpd.GeoDataFrame(csa, geometry='geometry', crs='epsg:4326')

# load homicide dataset
homicides = pd.read_csv('clean_data/homicides.csv')

# drop rows without coords for merge step
homicides_geo = homicides.dropna(subset=['lat', 'long']).copy()

# prepare coords for matching with geometry bounds
homicides_geo['geometry'] = homicides_geo.apply(
    lambda row: Point(row['long'], row['lat']), axis=1
)
homicides_gdf = gpd.GeoDataFrame(homicides_geo, geometry='geometry', crs='epsg:4326')

# join datasets on spatial data
homicides_with_csa = gpd.sjoin(
    homicides_gdf,
    csa_gdf,
    how='left',
    predicate='within'
)

# check for row inflation in case of boundary conflicts, and unmatched points
print(len(homicides_geo), len(homicides_with_csa))
print(homicides_with_csa['csa_20'].isna().sum())

# single unmatched row - note in writeup that one incident fell outside of CSA bounds, so was removed
homicides_with_csa_clean = homicides_with_csa.dropna(subset=['csa_20'])

# drop spatial join artifacts from merged dataset
homicides_with_csa = homicides_with_csa[['date', 'csa_20']].copy()

#  load VB dataset & merge with CSA on neighborhood
vb = pd.read_csv('clean_data/vb.csv')
csa = pd.read_csv('clean_data/csa.csv')

# function to match neighborhood label to substring, with error handling logic
def find_csa_row(neighborhood, csa_df, neighborhood_field='neighborhood'):
    if pd.isna(neighborhood):
        return None
    pattern = r'\b' + re.escape(neighborhood.strip()) + r'\b'
    match = csa_df[csa_df[neighborhood_field].str.contains(pattern, case=False, regex=True, na=False)]
    if len(match) == 1:
        return match.iloc[0]['csa_20']
    elif len(match) > 1:
        return 'Multiple Matches! Investigate'
    else:
        return None

vb['csa_20'] = vb['neighborhood'].apply(lambda n: find_csa_row(n, csa))

# check for unmatched or ambiguous rows
print(vb['csa_20'].isna().sum())
print((vb['csa_20'] == 'MULTIPLE_MATCH').sum())

# aggregate homicide incidents & VB counts by CSA
homicide_counts = homicides_with_csa.groupby('csa_20').size().reset_index(name='homicide_count')
vb_counts = vb.groupby('csa_20').size().reset_index(name='vacancy_count')

# load remaining datasets
income = pd.read_csv('clean_data/income.csv')
pop = pd.read_csv('clean_data/pop.csv')

# merge datasets as master - vb counts & homicides on csa_20, income on csa_20, population on csa_10 to bridge to csa_20
master = csa[['csa_20', 'csa_10', 'neighborhood']].drop_duplicates(subset='csa_20')
master = master.merge(vb_counts, on='csa_20', how='left')
master = master.merge(homicide_counts, on='csa_20', how='left')
master = master.merge(income, on='csa_20', how='left')
master = master.merge(pop, on='csa_10', how='left')

# replace any nulls with 0 for final analysis
master[['vacancy_count', 'homicide_count']] = master[['vacancy_count', 'homicide_count']].fillna(0)

# remove unnecessary columns & rename for final output
master = master[['vacancy_count', 'homicide_count', 'csa_20', 'area_sqft_x', '2023', 'pop']]
master = master.rename(columns={'area_sqft_x': 'area_sqft', 'csa_20': 'csa', '2023': 'income'})

# check that final dataset contains 55 entries and no nulls, check data types, and overall quality
print(len(master))
print(master.isna().sum())
print(master.dtypes)
print(master.describe())

# check that vacancy_count and homicide_count contain only whole numbers, then convert to int type for regression
print((master['vacancy_count'] % 1 == 0).all())
print((master['homicide_count'] % 1 == 0).all())
master['vacancy_count'] = master['vacancy_count'].astype(int)
master['homicide_count'] = master['homicide_count'].astype(int)

target_path = 'clean_data/master.csv'
master.to_csv(target_path, index=False)
print("Final dataset exported.")



