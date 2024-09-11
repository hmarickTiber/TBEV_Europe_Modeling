import re
import pandas as pd
from difflib import get_close_matches
from fuzzywuzzy import process, fuzz
import geopandas as gpd
from pyproj import Proj, transform
import config.paths as path

import sys

def fuzzy_apply(x, df, column, check, threshold=80):
    if type(x)!=str:
        return None

    if check is not None:
        match, score, *_ = process.extract(x, df[column], limit=1)[0]
        match_check, score_check, *_ = process.extract(x, df[check], limit=1, scorer=fuzz.ratio)[0]
            
        if score > score_check:
            if score > threshold:
                return match
            else:
                return None
        else:
            if score_check > threshold:
                    return match_check
            else:
                return None

    else: 
        match, score, *_ = process.extract(x, df[column], limit=1)[0]
            
        if score >= threshold:
            return match
        else:
            return None
        
def fuzzy_apply_score(x, df, column, check, threshold=80):
    if type(x)!=str:
        return None

    if check is not None:
        match, score, *_ = process.extract(x, df[column], limit=1)[0]
        match_check, score_check, *_ = process.extract(x, df[check], limit=1, scorer=fuzz.ratio)[0]
            
        if score > score_check:
            if score > threshold:
                return score
            else:
                return None
        else:
            if score_check > threshold:
                    return score_check
            else:
                return None

    else: 
        match, score, *_ = process.extract(x, df[column], limit=1)[0]
            
        if score >= threshold:
            return score
        else:
            return None

#define fuzzy df merge function
def fuzzy_merge(df, df2, on=None, left_on=None, right_on=None, check=None, how='inner', threshold=80):

    if on is not None:
        left_on = on
        right_on = on

    # create temp column as the best fuzzy match (or None!)
    df1 = df.copy()
    df1['tmp'] = df1[left_on].apply(
        fuzzy_apply, 
        df=df2, 
        column=right_on,
        check=check,
        threshold=threshold
    )
    df1['tmp_score'] = df1[left_on].apply(
        fuzzy_apply_score, 
        df=df2, 
        column=right_on,
        check=check,
        threshold=threshold
    )

    merged_df = df1.merge(df2, how=how, left_on='tmp', right_on=right_on)
    #del merged_df['tmp']
    
    return merged_df


def nuts_compiler(shapefile, country_codes):
    """ Return a tabular dataframe mapping all nuts levels for a specified country_code
    Requires specifically nuts level shapefile from europa:
    'https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/'
 
    Parameters
    ------------
        shapefile: string
            path to shapefile, including the filename.
            i.e.: "data/NUTS_RG_20M_2021_3035.shp/NUTS_RG_20M_2021_3035.shp"
        country_code: tuple of string 
            2 letter country code
            i.e. ('DE','NO') for germany
            NOTE: single quotes matter
    Return
    -----------
        geopandas dataframe.
    """

    #load in europa shp file to match on nuts_name and make lowercase
    gdf_3 = gpd.read_file(shapefile, where=f"CNTR_CODE IN {country_codes} AND LEVL_CODE=3")
    gdf_2 = gpd.read_file(shapefile, where=f"CNTR_CODE IN {country_codes} AND LEVL_CODE=2")
    gdf_1 = gpd.read_file(shapefile, where=f"CNTR_CODE IN {country_codes} AND LEVL_CODE=1")

    drop_cols = ['mount_type','urbn_type','coast_type','name_latn']

    gdf_3.columns = map(str.lower, gdf_3.columns)
    gdf_3 = gdf_3.applymap(lambda s: s.lower() if type(s) == str else s)
    gdf_3 = gdf_3.drop(drop_cols,axis=1).add_suffix('3')
    gdf_3 = gdf_3.set_geometry("geometry3")

    gdf_2.columns = map(str.lower, gdf_2.columns)
    gdf_2 = gdf_2.applymap(lambda s: s.lower() if type(s) == str else s)
    gdf_2 = gdf_2.drop(drop_cols,axis=1).add_suffix('2')
    gdf_2 = gdf_2.set_geometry("geometry2")

    gdf_1.columns = map(str.lower, gdf_1.columns)
    gdf_1 = gdf_1.applymap(lambda s: s.lower() if type(s) == str else s)
    gdf_1 = gdf_1.drop(drop_cols,axis=1).add_suffix('1')
    gdf_1 = gdf_1.set_geometry("geometry1")

    #make centroid column for nuts3
    gdf_nuts3 = gdf_3.copy()
    gdf_nuts3['centroid'] = gdf_nuts3['geometry3'].centroid
    gdf_nuts3['centroid_copy'] = gdf_nuts3['geometry3'].centroid #used for second merge
    gdf_nuts3 = gdf_nuts3.set_geometry("centroid")

    #merge nuts3 to nuts2
    merge_32 = gpd.sjoin(gdf_2, gdf_nuts3, how='left',predicate='intersects')
    merge_32 = merge_32.set_geometry("centroid_copy").drop('index_right',axis=1)

    #merge merge_32 to nuts1
    merged = gpd.sjoin(gdf_1, merge_32, how='left', predicate='intersects')
    
    return merged

#define function to convert from gps degree to gps decimal coords
def gps_convert(obj):
    pattern = r'(?P<d>[\d\.]+).*?(?P<m>[\d\.]+).*?(?P<s>[\d\.]+)'
    if type(obj) != str:
        return obj
    elif '°' in obj:
        if obj.endswith('N') or obj.endswith('E'):
            obj = obj[:-1] + '0' + obj[-1:]
            dms = re.split(pattern, obj)
            return float(dms[1]) + float(dms[2])/(60.0) + float(dms[3])/(3600.0)
        else:
            dms = re.split(pattern, obj)
            return float(dms[1]) + float(dms[2])/(60.0) + float(dms[3])/(3600.0)
    else:
        return obj
    
#define function to convert utm to latlong coords
def convert_utm_to_latlon(df, zone_number, zone_letter):
    utm_proj = Proj(proj='utm', zone=zone_number, ellps='WGS84', south=(zone_letter < 'N'))
    lonlat_proj = Proj(proj='latlong', datum='WGS84')
    
    lon, lat = transform(utm_proj, lonlat_proj, df['gps_e_imputed'].values, df['gps_n_imputed'].values)
    
    return pd.DataFrame({'longitude_raw': lon, 'latitude_raw': lat})

def fToC(deg): 
    return (deg-32)*5/9

def inch2mm(value):
    return 25.4*value

def nuts_join_regions(df):
    #add in NUTS3 level information by country

    #create geopandas points from lat/long coordinates and create gdf from df df
    if 'lon_env' in list(df.columns):
        df_geom = gpd.points_from_xy(df['lon_env'], df['lat_env'])
        df_gdf = gpd.GeoDataFrame(df,geometry=df_geom)
    else:
        df_geom = gpd.points_from_xy(df['longitude'], df['latitude'])
        df_gdf = gpd.GeoDataFrame(df,geometry=df_geom)

    df_merged_list = []

    #read in geopandas df for geo files from 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/'
    #read in nuts3 shapefile
    gdf_nuts = gpd.read_file("../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                        where=f"LEVL_CODE=3")
    gdf_nuts = gdf_nuts.to_crs(crs='epsg:4326')

    #read in master database subset by country_code
    current_gdf = df_gdf

    #perform merge of points to nuts3 shapefile
    merged_intersection = gpd.sjoin(current_gdf,gdf_nuts,how='left',predicate='intersects')
    merged_intersection = merged_intersection[~merged_intersection['NUTS_ID'].isnull()]

    #do a nearest merge for nan values using nuts3 shapefile and select those pts
    merged_closest = gpd.sjoin_nearest(current_gdf, gdf_nuts,how='left', distance_col='distance_from_boundary')
    merged_closest = merged_closest[merged_closest['distance_from_boundary'] != 0]

    #concatenate nuts3 dfs
    merged_nuts = pd.concat([merged_intersection, merged_closest]).sort_index(ascending=True)

    #process merged df, make everything lowercase
    merged_nuts.columns = map(str.lower, merged_nuts.columns)

    merged_nuts=merged_nuts.drop(['mount_type','urbn_type','coast_type','index_right','distance_from_boundary','name_latn','fid'],axis=1)

    return merged_nuts


##define function to join nuts3 and admin_locality names to dataframe 
def nuts_join(df):
    #add in NUTS3 level information by country
    #add in municipality level information by country (1 administrative level under nuts3)
    #make sure the dataframe has a country code! 

    #create geopandas points from lat/long coordinates and create gdf from df df
    if 'lon_env' in list(df.columns):
        df_geom = gpd.points_from_xy(df['lon_env'], df['lat_env'])
        df_gdf = gpd.GeoDataFrame(df,geometry=df_geom)
        df_gdf.set_crs('epsg:4326',inplace=True)
    else:
        df_geom = gpd.points_from_xy(df['longitude'], df['latitude'])
        df_gdf = gpd.GeoDataFrame(df,geometry=df_geom)
        df_gdf.set_crs('epsg:4326',inplace=True)

    df_merged_list = []

    #read in geopandas df for geo files from 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/'
    #read in geopandas df for administrative polygons from https://geodata.lib.utexas.edu/"
    for country_code in set(df_gdf['country_code']):
        print(country_code)
        #read in nuts3 shapefile
        gdf_nuts = gpd.read_file("../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                            where=f"CNTR_CODE='{country_code}' AND LEVL_CODE=3")
        gdf_nuts = gdf_nuts.to_crs(crs='epsg:4326')

        #read in master database subset by country_code
        current_gdf = df_gdf[df_gdf['country_code']==country_code]

        #perform merge of points to nuts3 shapefile
        merged_intersection = gpd.sjoin(current_gdf,gdf_nuts,how='left',predicate='intersects')
        merged_intersection = merged_intersection[~merged_intersection['NUTS_ID'].isnull()]

        #do a nearest merge for nan values using nuts3 shapefile and select those pts
        merged_closest = gpd.sjoin_nearest(current_gdf, gdf_nuts,how='left', distance_col='distance_from_boundary')
        merged_closest = merged_closest[merged_closest['distance_from_boundary'] != 0]

        #concatenate nuts3 dfs
        merged_nuts = pd.concat([merged_intersection, merged_closest]).sort_index(ascending=True)

        #process merged df, make everything lowercase
        merged_nuts.columns = map(str.lower, merged_nuts.columns)

        merged_nuts=merged_nuts.drop(['mount_type','urbn_type','coast_type','index_right','distance_from_boundary','cntr_code','name_latn','fid'],axis=1)
        df_merged_list.append(merged_nuts)

    gdf_proc = pd.concat(df_merged_list)
    gdf_proc = gdf_proc.to_crs('epsg:4326')
    return gdf_proc

##define function to join nuts3 and admin_locality names to dataframe 
def nuts_locality_join(df):
    #add in NUTS3 level information by country
    #add in municipality level information by country (1 administrative level under nuts3)
    #make sure the dataframe has a country code! 

    #create geopandas points from lat/long coordinates and create gdf from df df
    df_geom = gpd.points_from_xy(df['longitude'], df['latitude'])
    df_gdf = gpd.GeoDataFrame(df,geometry=df_geom)
    df_gdf.set_crs('epsg:4326',inplace=True)

    adm_shp_path_dict = path.adm_shp_path_dict

    df_merged_list = []

    #read in geopandas df for geo files from 'https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/'
    #read in geopandas df for administrative polygons from https://geodata.lib.utexas.edu/"
    for country_code in set(df_gdf['country_code']):
        print(country_code)
        #read in nuts3 shapefile
        gdf_nuts = gpd.read_file("../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                            where=f"CNTR_CODE='{country_code}' AND LEVL_CODE=3")
        gdf_nuts = gdf_nuts.to_crs(crs='epsg:4326')

        #read in administrative level shapefile
        gdf_adm = gpd.read_file(adm_shp_path_dict[country_code])
        gdf_adm = gdf_adm.to_crs(crs='epsg:4326')
        if country_code == 'EE':
            gdf_adm = gdf_adm[gdf_adm['NAME_1'] != 'Peipsi'] #for estonia get rid of giant lake region

        #read in master database subset by country_code
        current_gdf = df_gdf[df_gdf['country_code']==country_code]

        #perform merge of points to nuts3 shapefile
        merged_intersection = gpd.sjoin(current_gdf,gdf_nuts,how='left',predicate='intersects')
        merged_intersection = merged_intersection[~merged_intersection['NUTS_ID'].isnull()]

        #do a nearest merge for nan values using nuts3 shapefile and select those pts
        merged_closest = gpd.sjoin_nearest(current_gdf, gdf_nuts,how='left', distance_col='distance_from_boundary')
        merged_closest = merged_closest[merged_closest['distance_from_boundary'] != 0]

        #perform merge of points to adm shapefile
        merged_intersection_adm = gpd.sjoin(current_gdf,gdf_adm,how='left',predicate='intersects')
        merged_intersection_adm = merged_intersection_adm[~merged_intersection_adm['NAME_2'].isnull()]

        #do a nearest merge for nan values using adm shapefile and select those pts
        merged_closest_adm = gpd.sjoin_nearest(current_gdf, gdf_adm, how='left', distance_col='distance_from_boundary')
        merged_closest_adm = merged_closest_adm[merged_closest_adm['distance_from_boundary'] != 0]

        #concatenate nuts3 dfs
        merged_nuts = pd.concat([merged_intersection, merged_closest]).sort_index(ascending=True)
        merged_adm = pd.concat([merged_intersection_adm, merged_closest_adm]).sort_index(ascending=True)

        #concatenate adm dfs, with special conditions for germany=adm3 while others are adm2 for 1 lvl below NUTS3
        if 'adm2' in adm_shp_path_dict[country_code]:
            merged_adm.drop(merged_adm.columns.difference(['row_observation','NAME_1','NAME_2']), axis=1, inplace=True)
            merged = merged_nuts.merge(merged_adm, how='inner', left_on='row_observation', right_on='row_observation')
            merged.rename(columns={"NAME_1":"county_name", "NAME_2": "admin_name"}, inplace=True)

        else: 
            merged_adm.drop(merged_adm.columns.difference(['row_observation','NAME_1','NAME_3']), axis=1, inplace=True)
            merged = merged_nuts.merge(merged_adm, how='inner', left_on='row_observation', right_on='row_observation')
            merged.rename(columns={"NAME_1":"county_name", "NAME_3": "admin_name"}, inplace=True)

        #process merged df, make everything lowercase
        merged.columns = map(str.lower, merged.columns)

        merged=merged.drop(['mount_type','urbn_type','coast_type','index_right','distance_from_boundary','cntr_code','name_latn','fid'],axis=1)
        df_merged_list.append(merged)

    gdf_proc = pd.concat(df_merged_list)
    gdf_proc = gdf_proc.to_crs('epsg:4326')
    gdf_proc = gdf_proc.sort_values(by='row_observation', axis=0, ascending=True)

    return gdf_proc

def remove_or_combine_duplicates(df, strategy='remove', aggfunc=None):
    """
    Remove or aggregate duplicate columns in a DataFrame.

    :param df: DataFrame with potential duplicate columns.
    :param strategy: 'remove' to keep the first and remove other duplicates, 
                     'aggregate' to combine duplicate columns using aggfunc.
    :param aggfunc: Function used for aggregation when strategy='aggregate'. 
                    Must be a function acceptable by DataFrame.agg (e.g., 'mean', 'sum').
    :return: DataFrame with handled duplicates.
    """
    if strategy == 'remove':
        return df.loc[:, ~df.columns.duplicated()]
    elif strategy == 'aggregate' and aggfunc:
        return df.groupby(df.columns, axis=1).agg(aggfunc)
    else:
        raise ValueError("Invalid strategy or aggfunc is None when required.")