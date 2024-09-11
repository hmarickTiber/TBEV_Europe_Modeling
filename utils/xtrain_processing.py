import sys

import os
import utils.s3_utils as s3
import utils.processing as pr
import pandas as pd
import geopandas as gpd
import config.paths as path

def training_proc(country_code, scaler=20):
    
    #read in processed country predictor data
    # read_filename = f'../../data/env-clim-processed/buffered/{country_code}-predictors.parquet' #old pipeline
    read_filename = f'../../data/env-clim-processed-allrasters-allreservoir/buffered/{country_code}-predictors.parquet'
    pred_df = pd.read_parquet(read_filename)
    pred_df = gpd.GeoDataFrame(
        pred_df, geometry=gpd.points_from_xy(pred_df['lon_env'], pred_df['lat_env']))
    pred_df.set_crs('epsg:3035',inplace=True)

    bucket='pfizer-tbev'
    path_to_read_file='Germany/processed-data/processed_master_database/'
    write_filename='{country_code}_weather_landcover_master_processed.csv'
    read_filename = 'processed_master_database.csv'

    tick_df = s3.readCSVFromS3(bucket, path_to_read_file+read_filename).drop('Unnamed: 0', axis=1)
    tick_df = tick_df[tick_df['country_code'].isin([country_code])]

    tdf = tick_df[['latitude','longitude','obs_type', 'nuts_name', 'admin_name']]
    tdf = tdf.rename({'latitude': 'lat_foci', 'longitude':'lon_foci'},axis=1)

    #upsample microfocus data (double strength of  microfocus and confirmed microfocus data points)
    tdf = pd.concat([tdf,tdf[tdf['obs_type'].isin(['Confirmed Microfocus', 'Confirmed Natural Focus'])]])
    
    #label data for presence/absence
    tdf['presence'] = tdf['obs_type'].apply(lambda x: 0 if (x=='No focus') | (x=='No Focus') | (x=='Absent') else 1)

    #sjoin_nearest predictor data to TBE data.
    tick_gdf = gpd.GeoDataFrame(
        tdf, geometry=gpd.points_from_xy(tdf['lon_foci'], tdf['lat_foci']))
    tick_gdf.set_crs('epsg:3035',inplace=True)
    train_df = tick_gdf.sjoin_nearest(pred_df, how='left')
    train_df.columns = map(str.lower, train_df.columns)
    #train_df = train_df.drop_duplicates(keep='first')

    #write training data
    write_filename = f'../../data/training/training_data/{country_code}-training-data.csv'
    train_df.to_csv(write_filename)


    ### PSEUDOABSENCE POINTS ###

    # # read in landcover data cut out by unbuffered country shapes
    # read_filename = f'../../data/env-clim-processed/unbuffered/{country_code}-predictors.parquet' #old pipeline
    read_filename = f'../../data/env-clim-processed-allrasters-allreservoir/unbuffered/{country_code}-predictors.parquet'
    pred_df = pd.read_parquet(read_filename)
    pred_df = gpd.GeoDataFrame(
        pred_df, geometry=gpd.points_from_xy(pred_df['lon_env'], pred_df['lat_env']))
    pred_df.set_crs('epsg:3035',inplace=True)

    # #get subset of env data without tick data points,
    env_no_foci = pred_df[~(pred_df['geometry'].isin(list(train_df['geometry'])))]

    #use stratified sampling to get all landcover category values
    #Fix n to be ~N=1000 for norway, but scale to size of other countries
    # n_sample = len(env_no_foci)/30000
    #Another idea to try - scale by # of data observations we have per country (10x obs points)
    n_sample = len(train_df)*scaler

    print(f'{n_sample} Pseudoabsence points Generated for {country_code}')
    pseudoabsence = env_no_foci.groupby('cat', group_keys=False).apply(lambda x: x.sample(frac=(n_sample/len(env_no_foci))))

    #make sure to get at least 1 sample of every landcover type by using the set diff between pred_df and pseudoabsence
    for landcover in set(pred_df['cat']).difference(set(pseudoabsence['cat'])):
        pa_strat_sample = env_no_foci[env_no_foci['cat']==landcover].sample(1)
        pseudoabsence = pd.concat([pseudoabsence, pa_strat_sample])
    pseudoabsence['presence'] = 0

    ### map nuts and admin localities to pseudoabsence points ###
    adm_shp_path_dict = path.adm_shp_path_dict


    #read in nuts3 shapefile
    gdf_nuts = gpd.read_file("../../data/shapefiles/NUTS_RG_20M_2021_3035.shp/NUTS_RG_20M_2021_3035.shp", 
                        where=f"CNTR_CODE='{country_code}' AND LEVL_CODE=3")
    #gdf_nuts = gdf_nuts.to_crs(crs='epsg:3035')

    #read in administrative level shapefile
    gdf_adm = gpd.read_file(adm_shp_path_dict[country_code])
    gdf_adm = gdf_adm.to_crs(crs='epsg:3035')

    #get pseudoabsence gdf
    current_gdf = pseudoabsence.copy()

    ### Merge points to nuts3 and adm_district names. 
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

    #concatenate nuts3 and admin dfs
    merged_nuts = pd.concat([merged_intersection, merged_closest]).sort_index(ascending=True)
    merged_adm = pd.concat([merged_intersection_adm, merged_closest_adm]).sort_index(ascending=True)

    #concatenate adm dfs, with special conditions for germany=adm3 while others are adm2 for 1 lvl below NUTS3
    if 'adm2' in adm_shp_path_dict[country_code]:
        merged_adm.drop(merged_adm.columns.difference(['row_observation','NAME_1','NAME_2']), axis=1, inplace=True)
        merged = merged_nuts.merge(merged_adm, how='inner', left_index=True, right_index=True)
        merged.rename(columns={"NAME_1":"county_name", "NAME_2": "admin_name"}, inplace=True)

    else: 
        merged_adm.drop(merged_adm.columns.difference(['row_observation','NAME_1','NAME_3']), axis=1, inplace=True)
        merged = merged_nuts.merge(merged_adm, how='inner', left_index=True, right_index=True)
        merged.rename(columns={"NAME_1":"county_name", "NAME_3": "admin_name"}, inplace=True)

    #process merged df, make everything lowercase
    merged.columns = map(str.lower, merged.columns)

    pseudo_merged=merged.drop(['mount_type','urbn_type','coast_type','index_right','distance_from_boundary','name_latn','fid','nuts_id','county_name','levl_code'],axis=1)
    pseudo_merged=pseudo_merged.rename({'cntr_code':'country_code'},axis=1)

    write_filename = f'../../data/training/pseudoabsence_data/{country_code}-pseudoabsence-data.csv'
    pseudo_merged.to_csv(write_filename)

    #return train_df, pseudo_merged
