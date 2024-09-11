import sys
import os
import rasterio
import config.paths as path
import folium
import branca
import itertools
from folium.elements import Element
from jinja2 import Template
import re
import branca.colormap as cmp
import pandas as pd
import geopandas as gpd
import shapefile as shp
from shapely.geometry import Point
from shapely.geometry.polygon import Point, Polygon
import matplotlib.pyplot as plt
import rioxarray as rxr
import xarray as xr
import numpy as np
import utils.processing as pr
import utils.s3_utils as s3
import config.paths as path

modelname = path.model_path

width = 6000
height = 6000
zoom_level=7

### For larger maps:
# width = 1800
# height = 1800
# zoom_level=5

def bg_changer(m, color='white'):
    ###add css to change bg color to white
    map_id = m._id
    print(map_id)
    bg_css = "<style>#map_" + str(map_id) + """
        {
            background-color: """ + color + """
        }
    </style>
    """
    Element(bg_css).add_to(m.get_root().header)
    return m

def plot_log_maps(df, cntry_codes, region='n', dot_sample=40000):
    ## Must run this for ANY map production, even choropleth maps!!
    ctry_map = path.ctry_map

    country_codes = cntry_codes

    ## all localities in one gdf
    shp_nuts = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=3")
    shp_nuts = shp_nuts[shp_nuts['CNTR_CODE'].isin(country_codes)]
    shp_nuts = shp_nuts.to_crs(crs='epsg:4326')

    shp_gdf = shp_nuts.copy()

    shp_cntry = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=0")
    shp_cntry = shp_cntry[shp_cntry['CNTR_CODE'].isin(country_codes)]
    shp_cntry = shp_cntry.to_crs(crs='epsg:4326')

    # get list of geometries
    geoms = shp_cntry['geometry'].tolist()

    # iterate over all combinations of polygons and get the intersections (overlaps)
    overlaps = gpd.GeoDataFrame(gpd.GeoSeries([poly[0].intersection(poly[1]) for poly in itertools.combinations(geoms, 2) if poly[0].intersects(poly[1])]), columns=['geometry'])

    # set the crs
    overlaps.crs = shp_cntry.crs
    shp_cntry1 = shp_cntry.overlay(overlaps, how='difference')


    ###grab processed tick data to plot
    #grab processed tick data from local
    path_to_read_file=f'../../data/{modelname}/processed-master-database/'
    file_name_microfoci = 'cluster_df.csv'
    tick_df = pd.read_csv(path_to_read_file+file_name_microfoci)

    gdf = gpd.GeoDataFrame(
        tick_df, geometry=gpd.points_from_xy(tick_df['longitude'], tick_df['latitude']))
    gdf = gdf.drop(['Unnamed: 0','index_right'],axis=1)

    if region == 'n':
        df_foci = tick_df[tick_df['country_code'].isin(country_codes)]
        print('region==False')
    
    else:
        ### need to do the nuts3 filtering below after choropleth
        # keep_nuts3 = list(set(df_foci.NUTS_ID))
        # shp_gdf = shp_gdf[shp_gdf.NUTS_ID.isin(keep_nuts3)]

    
        shp_cntry = gpd.read_file(f"../../data/{modelname}/geographic-clustering/region-shp-polygons/region_{region}.shp")

        shp_cntry1 = shp_cntry.drop([col for col in shp_cntry.columns if col !='geometry'], axis=1)


        df_foci = gpd.sjoin(gdf,shp_cntry1, how='inner',predicate='intersects')
        keep_countries = list(set(df_foci.country_code))

    ### Begin plotting in folium
    # location = [59, 14.5]
    location = [df_foci.latitude.mean(), df_foci.longitude.mean()]


    f = folium.Figure(width=width, height=height)
    m = folium.Map(location=location, zoom_start=4, 
                tiles=None,
                width=width, height=height,
                control_scale=False,
                zoom_control=False,
                min_zoom=2,
                max_zoom=15,
                zoomDelta=0.5,
                zoomSnap=0.5, #Faster debounce time while zooming
                wheelDebounceTime=20,
                wheelPxPerZoomLevel=20
                #    scrollWheelZoom=False, # disable original zoom function
                #    smoothWheelZoom=True,  # enable smooth zoom 
                #    smoothSensitivity=1   # zoom speed. default is 1
                ).add_to(f)

        # colors=['#bbface', '#26ff80', '#029c42', 'green', 'yellow', 'orange', 'red'],
        # index=[0,10,20,35,55,60,80],

    linear = branca.colormap.StepColormap(
        colors=['#D0F0D4', '#00ab47', 'yellow','orange','red', '#910101'],
        index=[0,5,10,20,30,40],
        vmin=0, vmax=100,
        caption='TBEV Foci Probability (%)' #Caption for Color scale or Legend
    )

    ### below is the same code for maxent output
    # linear = branca.colormap.StepColormap(
    #     colors=['#D0F0D4', '#00ab47', 'green', 'yellow','orange','red', '#910101'],
    #     index=[0,10,20,30,45,60,80],
    #     vmin=0, vmax=100,
    #     caption='TBEV Foci Probability (%)' #Caption for Color scale or Legend
    # )
    # linear.add_to(m)

    predictions = folium.FeatureGroup('Heatmap')
    # #heatmap method
    # HeatMap(
    #     test_df[['lat','lon','pred']],
    #     min_opacity=0.3, radius=25, blur=15, overlay=True, control=True, show=True,
    #     gradient={0.7: 'blue', 0.9: 'lime', 0.91: 'yellow', 0.92: 'orange', .94: 'red'}).add_to(predictions)
    # predictions.add_to(m)

    #sampling points method:
    test_sample = df.sample(dot_sample)
    test_sample['proba'] = np.round(test_sample.pred*100)

    test_sample = gpd.GeoDataFrame(
        test_sample, geometry=gpd.points_from_xy(test_sample['longitude'], test_sample['latitude']))
    test_sample = test_sample[['geometry','proba']]

    #get test_sample vals inside valid countries (remove pts outside countries but inside regions)
    test_sample = test_sample.overlay(shp_cntry1, how='intersection')

    for _, r in test_sample.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                zoom_on_click=False,
                                tooltip = f"Probability: {str(r['proba'])}%",
                                smooth_factor=1,
                                marker=folium.Circle(
                                    radius=2300,
                                    fill_color=linear(r['proba']), 
                                    fill_opacity=0.5, 
                                    color=linear(r['proba']), 
                                    weight=1,
                                    opacity = .2
                                ))
        geo_j.add_to(predictions)
    predictions.add_to(m)

    ### plot countries, localities uses choropleth filtering below (Or using region)
    fg_base = folium.FeatureGroup(name='countries')
    for _, r in shp_cntry1.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(.0001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            #tooltip=f"<b>District: {r['NAME_2']}",
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .5,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)


    ### plot tbe data
    df_foci['obs_type'] = np.where(df_foci['presence'] == 1, 'Focus', 'No Focus')
    gdf = gpd.GeoDataFrame(
        df_foci[['obs_type','tick_animal']], geometry=gpd.points_from_xy(df_foci['longitude'], df_foci['latitude']))

    ##################################################################################################

    # prev version coloring foci pts by region
    presence_colors = {
        'Tick' : 'black',
        'Rodent reservoir' : '#500C46',
    }

    presence_outlines = {
        'Tick' : 'black',
        'Rodent reservoir' : 'black'
    }

    c_rad=2.5
    fill_opac=.9
    opac=.8
    line_wt=.15

    mft = folium.FeatureGroup('foci_tick')
    for _, r in gdf[gdf['tick_animal']=='Tick'].iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                zoom_on_click=False,
                                tooltip =  f"Obs. Type: {r['animal_category']}",
                                marker=folium.CircleMarker(
                                radius=c_rad,
                                fill_color=presence_colors[r['animal_category']], 
                                fill_opacity=fill_opac, 
                                color=presence_outlines[r['tick_animal']],
                                opacity=opac,
                                weight=line_wt
                                ))
        geo_j.add_to(mft)
    mft.add_to(m)

    t_rad = .22
    mfa = folium.FeatureGroup('foci_animal')
    for _, r in gdf[gdf['tick_animal']=='Animal'].iterrows():
        triangle = [
        [r['latitude']+t_rad*1,r['longitude']],  # upper point
        [r['latitude']-t_rad*1,r['longitude']-t_rad*1.7],  # left point
        [r['latitude']-t_rad*1,r['longitude']+t_rad*1.7]]   # right point
        
        folium.Polygon(locations=triangle, 
                    fill=True, 
                    fill_color=presence_colors[r['animal_category']], 
                    fill_opacity=fill_opac, 
                    color=presence_outlines[r['tick_animal']],
                    opacity=opac,
                    weight=line_wt
                    ).add_to(mfa)
    mfa.add_to(m)


    # Define your HTML content for the legend
    legend_html = '''
    <div style="position: fixed; 
                top: 11px; left: 50px; width: 110px; 
                background-color: white; border:1px solid black; z-index:9999; 
                font-size:14px; display: flex; flex-direction: column; 
                align-items: center;  padding: 5px;">
        <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Foci Probability (%) </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #910101; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">40%+</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: red; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">30%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: orange; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">20%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: yellow; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">10%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #00ab47; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">5%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #D0F0D4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">0%</span>
        </div>
        <!-- Circle Marker for "TBE Focus" -->
        <div style="display: flex; width: 100%; align-items: center; margin-top: 10px;">
            <div style="margin-left: 5px; background-color: black; width: 10px; height: 10px; border-radius: 50%;"></div>
            <span style="font-size:10px; margin-left: 7px;">Tick</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-top: 10px;">
            <div style="margin-left: 5px;font-size:12px;color:#500C46">&#9650;</div>
            <span style="font-size:10px; margin-left: 7px;">Rodent reservoir</span>
        </div>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)

    #folium.LayerControl().add_to(m)
    m.get_root().add_child(legend)


    ### NEXT SECTION IS FOR CHOROPLETH MAP

    ### map NUTS3 localities to prediction points ###
    pred_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])).reset_index(drop=True)

    merged_intersection_adm = gpd.sjoin(pred_gdf,shp_gdf,how='left',predicate='intersects')

    #get missed rows
    leftovers = merged_intersection_adm[merged_intersection_adm['NUTS_ID'].isnull()][['latitude','longitude','pred','geometry']]

    #save properly merged rows
    merged_intersection_adm = merged_intersection_adm[~merged_intersection_adm['NUTS_ID'].isnull()]

    #nearest join with leftover points
    merged_closest_adm= gpd.sjoin_nearest(leftovers, shp_gdf, how='left', distance_col='distance_from_boundary')

    #concatenate intersected and nearest dfs
    merged_adm = pd.concat([merged_intersection_adm, merged_closest_adm]).reset_index(drop=True).drop('distance_from_boundary',axis=1)

    #merged_adm = merged_adm[['index_right','longitude','latitude','pred','NUTS_ID','NUTS_NAME','CNTR_CODE','geometry']]
    
    #drop any districts with no predictions
    merged_adm = merged_adm.dropna(subset=['pred'])

    df_map = merged_adm.groupby(['index_right','CNTR_CODE','NUTS_ID'])['pred'].mean().reset_index()
    df_map['Probability'] = df_map.pred*100
    df_map['Probability'] = df_map['Probability'].round(1)
    df_map['unique_id_str'] = df_map.index.astype('str')
    df_map['District_Name'] = df_map['NUTS_ID']# +': ' +df_map['NUTS_NAME']
    gdf_map = shp_gdf.merge(df_map,how='inner',on=['NUTS_ID']).reset_index()
    gdf_map['geometry'] = gdf_map['geometry'].simplify(.001)
    gdf_map = gdf_map.to_crs('epsg:4326')


    gdf_map = gdf_map[['pred','NUTS_ID','NUTS_NAME','CNTR_CODE_x','geometry','District_Name','Probability']]
    gdf_map= gdf_map.rename({'NUTS_NAME':'Nuts3 District'},axis=1)
    #del merged_adm, merged_closest_adm, merged_intersection_adm 



    ### output files to shapefile for further editing
    # gdf_map.to_file('prediction_by_NUTS3')

        # location = [59, 14.5]
        # location = [64, 18]
    location = [df_foci.latitude.mean(), df_foci.longitude.mean()]

    ### create choropleth map 'ma'
    #map configs to adjust zoom settings
    fig = folium.Figure(width=width, height=height)
    ma = folium.Map(location=location, zoom_start=4, tiles=None, width=width, height=height,
                control_scale=False,
                zoom_control=False,
                min_zoom=2,
                max_zoom=15,
                zoomDelta=0.5,
                zoomSnap=0.25, #Faster debounce time while zooming
                wheelDebounceTime=20,
                wheelPxPerZoomLevel=20
                #    scrollWheelZoom=False, # disable original zoom function
                #    smoothWheelZoom=True,  # enable smooth zoom 
                #    smoothSensitivity=1   # zoom speed. default is 1
                ).add_to(fig)

    binning = [0,5,10,20,35,100]
    ### Add in custom choropleth legend
    # Define your custom color palette as a LinearColormap
    custom_palette = cmp.LinearColormap(
        colors=['#FEF0D9', '#FDD49E', '#FDBB84',
                '#FC8D59', '#EF6548', '#D7301F', '#990000'],  # Orange to red
        index=binning,
        vmin=0,
        vmax=100,
        caption='# of Total TBE Cases'
    )

    def style_function(feature):
        prob = feature['properties']['Probability']
        return {
            'fillColor': custom_palette(prob),
            'color': 'black',  # Line color
            'weight': 0.25,  # Line width
            'fillOpacity': 0.95
        }


    #create choropleth map
    cp = folium.Choropleth(
        geo_data = gdf_map.to_json(drop_id=True),
        name = 'choropleth',
        data = gdf_map,
        style_function = style_function,
        columns = ("District_Name","Probability"),
        key_on = "feature.properties.District_Name",
        fill_color = 'OrRd',
        bins=binning,
        nan_fill_color = 'grey',
        fill_opacity = 0.95,
        line_opacity = 1,
        line_weight=.25,
        legend_name = 'Foci Probability (%)'
    ).add_to(ma)

    # creating a state indexed version of the dataframe so we can lookup values
    data_indexed = gdf_map.set_index('Nuts3 District')

    # looping thru the geojson object and adding a new property
    # and assigning a value from our dataframe

    for idx,s in enumerate(cp.geojson.data['features']):
        s['properties']['Probability'] = str(data_indexed.loc[s['properties']['Nuts3 District'], 'Probability'])+'%'

    ##and finally adding a tooltip/hover to the choropleth's geojson
    folium.GeoJsonTooltip(['Nuts3 District', 'Probability']).add_to(cp.geojson)

    # folium.GeoJson(
    #     gdf_map.to_json(drop_id=True),
    #     style_function=style_function,
    #     tooltip=folium.GeoJsonTooltip(fields=['Nuts3 District', 'Probability'], aliases=['District:', 'Probability:'])
    # ).add_to(ma)


    # Define your HTML content for the legend
    legend_html = '''
    <div style="position: fixed; 
                top: 11px; left: 50px; width: 110px; 
                background-color: white; border:1px solid black; z-index:9999; 
                font-size:14px; display: flex; flex-direction: column; 
                align-items: center;  padding: 5px;">
        <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Foci Probability (%) </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #990000; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">35%+</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #D7301F; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">20%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #FC8D59; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">10%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #FDBB84; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">5%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #FEF0D9; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">0</span>
        </div>
        <!-- Circle Marker for "TBE Focus" -->
        <div style="display: flex; width: 100%; align-items: center; margin-top: 10px;">
            <div style="margin-left: 5px; background-color: black; width: 10px; height: 10px; border-radius: 50%;"></div>
            <span style="font-size:10px; margin-left: 7px;">Tick</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-top: 10px;">
            <div style="margin-left: 5px;font-size:12px;color:#500C46">&#9650;</div>
            <span style="font-size:10px; margin-left: 7px;">Rodent reservoir</span>
        </div>
    </div>
    </div>
    '''
    ma.get_root().html.add_child(folium.Element(legend_html))


    ### Add foci data ###
    #foci data taken from above dotmap section

    presence_colors = {
        'Tick' : 'black',
        'Rodent reservoir' : '#500C46',
    }

    presence_outlines = {
        'Tick' : 'black',
        'Rodent reservoir' : 'black'
    }

    c_rad=2.5
    fill_opac=.8
    opac=.8
    line_wt=.15

    mft = folium.FeatureGroup('foci_tick')
    for _, r in gdf[gdf['tick_animal']=='Tick'].iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                zoom_on_click=False,
                                tooltip =  f"Obs. Type: {r['animal_category']}",
                                marker=folium.CircleMarker(
                                radius=c_rad,
                                fill_color=presence_colors[r['animal_category']], 
                                fill_opacity=fill_opac, 
                                color=presence_outlines[r['tick_animal']],
                                opacity=opac,
                                weight=line_wt
                                ))
        geo_j.add_to(mft)
    mft.add_to(m)

    t_rad = .22
    mfa = folium.FeatureGroup('foci_animal')
    for _, r in gdf[gdf['tick_animal']=='Animal'].iterrows():
        triangle = [
        [r['latitude']+t_rad*1,r['longitude']],  # upper point
        [r['latitude']-t_rad*1,r['longitude']-t_rad*1.7],  # left point
        [r['latitude']-t_rad*1,r['longitude']+t_rad*1.7]]   # right point
        
        folium.Polygon(locations=triangle, 
                    fill=True, 
                    fill_color=presence_colors[r['animal_category']], 
                    fill_opacity=fill_opac, 
                    color=presence_outlines[r['tick_animal']],
                    opacity=opac,
                    weight=line_wt
                    ).add_to(mfa)
    mfa.add_to(m)


    ### USE CHOROPLETH FILTERING FOR NUTS3 for map m (dotmap)

    ### plot localities
    fg_base = folium.FeatureGroup(name='NUTS3 Districts')
    for _, r in gdf_map.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'].simplify(.001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .1,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)


    return m, ma


def plot_xgboost_maps(df, cntry_codes, region='n', dot_sample=40000):
    ## Must run this for ANY map production, even choropleth maps!!
    ctry_map = path.ctry_map

    country_codes = cntry_codes

    ## all localities in one gdf
    shp_nuts = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=3")
    shp_nuts = shp_nuts[shp_nuts['CNTR_CODE'].isin(country_codes)]
    shp_nuts = shp_nuts.to_crs(crs='epsg:4326')

    shp_gdf = shp_nuts.copy()

    shp_cntry = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=0")
    shp_cntry = shp_cntry[shp_cntry['CNTR_CODE'].isin(country_codes)]
    shp_cntry = shp_cntry.to_crs(crs='epsg:4326')

    # get list of geometries
    geoms = shp_cntry['geometry'].tolist()

    # iterate over all combinations of polygons and get the intersections (overlaps)
    overlaps = gpd.GeoDataFrame(gpd.GeoSeries([poly[0].intersection(poly[1]) for poly in itertools.combinations(geoms, 2) if poly[0].intersects(poly[1])]), columns=['geometry'])

    # set the crs
    overlaps.crs = shp_cntry.crs
    shp_cntry1 = shp_cntry.overlay(overlaps, how='difference')

    ###grab processed tick data to plot
    #grab processed tick data from local
    path_to_read_file=f'../../data/{modelname}/processed-master-database/'
    file_name_microfoci = 'cluster_df.csv'
    tick_df = pd.read_csv(path_to_read_file+file_name_microfoci)

    gdf = gpd.GeoDataFrame(
        tick_df, geometry=gpd.points_from_xy(tick_df['longitude'], tick_df['latitude']))
    gdf = gdf.drop(['Unnamed: 0','index_right'],axis=1)

    if region == 'n':
        df_foci = tick_df[tick_df['country_code'].isin(country_codes)]
        print('region==False')
    
    else:
        ### need to do the nuts3 filtering below after choropleth
        # keep_nuts3 = list(set(df_foci.NUTS_ID))
        # shp_gdf = shp_gdf[shp_gdf.NUTS_ID.isin(keep_nuts3)]

        shp_cntry = gpd.read_file(f"../../data/{modelname}/geographic-clustering/region-shp-polygons/region_{region}.shp")
        shp_cntry1 = shp_cntry.drop([col for col in shp_cntry.columns if col !='geometry'], axis=1)

        df_foci = gpd.sjoin(gdf,shp_cntry1, how='inner',predicate='intersects')
        keep_countries = list(set(df_foci.country_code))

    ### Begin plotting in folium
    # location = [59, 14.5]
    location = [df_foci.latitude.mean()+4.7, df_foci.longitude.mean()-1.8]

    f = folium.Figure(width=width, height=height)
    m = folium.Map(location=location, zoom_start=zoom_level, 
                tiles=None,
                width=width, height=height,
                control_scale=False,
                zoom_control=False,
                min_zoom=2,
                max_zoom=15,
                zoomDelta=0.5,
                zoomSnap=0.5, #Faster debounce time while zooming
                wheelDebounceTime=20,
                wheelPxPerZoomLevel=20
                #    scrollWheelZoom=False, # disable original zoom function
                #    smoothWheelZoom=True,  # enable smooth zoom 
                #    smoothSensitivity=1   # zoom speed. default is 1
                ).add_to(f)
    
    #change bg color to white
    m = bg_changer(m,'white')

        # colors=['#bbface', '#26ff80', '#029c42', 'green', 'yellow', 'orange', 'red'],
        # index=[0,10,20,35,55,60,80],

    linear = branca.colormap.StepColormap(
        colors=['#f4f4f4','#D0F0D4', '#fde725','#5ec962','#21918c','#3b528b','#440154'], #viridis colors from https://waldyrious.net/viridis-palette-generator/
        index=[0,1,5,10,15,25,35],
        vmin=0, vmax=100,
        caption='TBEV Foci Probability (%)' #Caption for Color scale or Legend
    )

    predictions = folium.FeatureGroup('Heatmap')

    #sampling points method:
    test_sample = df.sample(dot_sample)
    test_sample['proba'] = np.round(test_sample.pred*100)

    test_sample = gpd.GeoDataFrame(
        test_sample, geometry=gpd.points_from_xy(test_sample['longitude'], test_sample['latitude']))
    test_sample = test_sample[['geometry','proba']]

    #get test_sample vals inside valid countries
    test_sample = test_sample.overlay(shp_cntry1, how='intersection')

    for _, r in test_sample.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                zoom_on_click=False,
                                tooltip = f"Probability: {str(r['proba'])}%",
                                smooth_factor=1,
                                marker=folium.Circle(
                                    radius=2300,
                                    fill_color=linear(r['proba']), 
                                    fill_opacity=0.5, 
                                    color=linear(r['proba']), 
                                    weight=1,
                                    opacity = .2
                                ))
        geo_j.add_to(predictions)
    predictions.add_to(m)

    ### plot countries, localities uses choropleth filtering below (Or using region)
    fg_base = folium.FeatureGroup(name='countries')
    for _, r in shp_cntry1.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(.0001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            #tooltip=f"<b>District: {r['NAME_2']}",
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .4,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)


    # Define your HTML content for the legend
    legend_html = '''
    <div style="position: fixed; 
                top: 11px; left: 50px; width: 120px; 
                background-color: white; border:1px solid black; z-index:9999; 
                font-size:14px; display: flex; flex-direction: column; 
                align-items: center;  padding: 5px;">
        <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Probability TBEV occurrence </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #440154; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">35%+</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #3b528b; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">25%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #21918c; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">15%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #5ec962; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">10%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #fde725; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">5%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #D0F0D4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">1%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #f4f4f4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">0%</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)

    #folium.LayerControl().add_to(m)
    m.get_root().add_child(legend)


    for key in m._children:
        if key.startswith('color_map'):
            print(key)
            del m._children[key]


    ### NEXT SECTION IS FOR CHOROPLETH MAP

    ### map NUTS3 localities to prediction points ###
    pred_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])).reset_index(drop=True)

    merged_intersection_adm = gpd.sjoin(pred_gdf,shp_gdf,how='left',predicate='intersects')

    #get missed rows
    leftovers = merged_intersection_adm[merged_intersection_adm['NUTS_ID'].isnull()][['latitude','longitude','pred','geometry']]

    #save properly merged rows
    merged_intersection_adm = merged_intersection_adm[~merged_intersection_adm['NUTS_ID'].isnull()]

    #nearest join with leftover points
    merged_closest_adm= gpd.sjoin_nearest(leftovers, shp_gdf, how='left', distance_col='distance_from_boundary')

    #concatenate intersected and nearest dfs
    merged_adm = pd.concat([merged_intersection_adm, merged_closest_adm]).reset_index(drop=True).drop('distance_from_boundary',axis=1)

    #merged_adm = merged_adm[['index_right','longitude','latitude','pred','NUTS_ID','NUTS_NAME','CNTR_CODE','geometry']]
    
    #drop any districts with no predictions
    merged_adm = merged_adm.dropna(subset=['pred'])

    df_map = merged_adm.groupby(['index_right','CNTR_CODE','NUTS_ID'])['pred'].mean().reset_index()
    df_map['Probability'] = df_map.pred*100
    df_map['Probability'] = df_map['Probability'].round(1)
    df_map['unique_id_str'] = df_map.index.astype('str')
    df_map['District_Name'] = df_map['NUTS_ID']# +': ' +df_map['NUTS_NAME']
    gdf_map = shp_gdf.merge(df_map,how='inner',on=['NUTS_ID']).reset_index()
    gdf_map['geometry'] = gdf_map['geometry'].simplify(.001)
    gdf_map = gdf_map.to_crs('epsg:4326')


    gdf_map = gdf_map[['pred','NUTS_ID','NUTS_NAME','CNTR_CODE_x','geometry','District_Name','Probability']]
    gdf_map= gdf_map.rename({'NUTS_NAME':'Nuts3 District'},axis=1)
    #del merged_adm, merged_closest_adm, merged_intersection_adm 
    gdf_map.to_csv("full-choropleth-pred.csv")

    ### create choropleth map 'ma'
    #location set in dotmap above
    #map configs to adjust zoom settings
    fig = folium.Figure(width=width, height=height)
    ma = folium.Map(location=location, zoom_start=zoom_level, tiles=None, width=width, height=height,
                control_scale=False,
                zoom_control=False,
                min_zoom=2,
                max_zoom=15,
                zoomDelta=0.5,
                zoomSnap=0.25, #Faster debounce time while zooming
                wheelDebounceTime=20,
                wheelPxPerZoomLevel=20
                #    scrollWheelZoom=False, # disable original zoom function
                #    smoothWheelZoom=True,  # enable smooth zoom 
                #    smoothSensitivity=1   # zoom speed. default is 1
                ).add_to(fig)
    
    #change bg color to white
    ma = bg_changer(ma,'white')

    #['#D0F0D4','#fde725','#5ec962','#21918c','#3b528b','#440154'] #viridis
    binning = [0,1,5,10,15,25,35,100]
    ### Add in custom choropleth legend
    # Define your custom color palette as a LinearColormap
    #https://github.com/python-visualization/folium/issues/403
    custom_palette = cmp.StepColormap(
        ['#f4f4f4','#D0F0D4','#fde725','#5ec962','#21918c','#3b528b','#440154'],#,'#440154'], #viridis w less opaque
       #['#E3EBF4', '#A6BDDB','#74A9CF', '#0570B0','#034E7B'],  # PuBu
        index=binning,
        vmin=0,
        vmax=100,
        caption='# of Total TBE Cases'
    )

    #create choropleth map
    cp = folium.Choropleth(
        geo_data = gdf_map.to_json(drop_id=True),
        name = 'choropleth',
        data = gdf_map,
        columns = ("District_Name","Probability"),
        key_on = "feature.properties.District_Name",
        bins=binning,
        nan_fill_color = 'grey',
        line_opacity = 1,
        line_weight=.1,
        legend_name = 'Foci Probability (%)'
    )

    ### delete colormap legends
    for key in cp._children:
        if key.startswith('color_map'):
            print(key)
            del cp._children[key]

    # creating a state indexed version of the dataframe so we can lookup values
    data_indexed = gdf_map.set_index('Nuts3 District')

    df_dict = gdf_map.set_index('District_Name')['Probability']
    style_function = lambda x: {'fillColor': custom_palette(df_dict[x['properties']['District_Name']]),
                                'color': 'black',  # Line color
                                'weight': 0.1,  # Line width
                                'fillOpacity': 0.85
        }
    
    cp.geojson.style_function = style_function
    cp.add_to(ma)

    # looping thru the geojson object and adding a new property
    # and assigning a value from our dataframe

    for idx,s in enumerate(cp.geojson.data['features']):
        s['properties']['Probability1'] = str(data_indexed.loc[s['properties']['Nuts3 District'], 'Probability'])+'%'

    ##and finally adding a tooltip/hover to the choropleth's geojson
    folium.GeoJsonTooltip(['Nuts3 District', 'Probability1']).add_to(cp.geojson)



    # #['#E3EBF4', '#A6BDDB','#74A9CF', '#3690C0', '#034E7B'] #actual blues palette
    # ['#D0F0D4','#5ec962','#21918c','#3b528b','#440154']
    # Define your HTML content for the legend
    legend_html = '''
    <div style="position: fixed; 
                top: 11px; left: 50px; width: 120px; 
                background-color: white; border:1px solid black; z-index:9999; 
                font-size:14px; display: flex; flex-direction: column; 
                align-items: center;  padding: 5px;">
        <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Probability TBEV occurrence </div>
                <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #440154; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">35%+</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #3b528b; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">25%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #21918c; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">15%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #5ec962; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">10%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #fde725; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">5%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #D0F0D4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">1%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #f4f4f4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">0%</span>
        </div>
    </div>
    '''
    ma.get_root().html.add_child(folium.Element(legend_html))


    ### USE CHOROPLETH FILTERING FOR NUTS3 for map m (dotmap)
    ### plot localities
    fg_base = folium.FeatureGroup(name='NUTS3 Districts')
    for _, r in gdf_map.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'].simplify(.001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .1,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)

    ### plot countries, localities uses choropleth filtering below (Or using region)
    fg_base = folium.FeatureGroup(name='countries')
    for _, r in shp_cntry1.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(.0001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            #tooltip=f"<b>District: {r['NAME_2']}",
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .4,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(ma)


    return m, ma




def plot_maxent_maps(df, cntry_codes, region='n', dot_sample=40000):
    ## Must run this for ANY map production, even choropleth maps!!
    ctry_map = path.ctry_map

    country_codes = cntry_codes

    ## all localities in one gdf
    shp_nuts = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=3")
    shp_nuts = shp_nuts[shp_nuts['CNTR_CODE'].isin(country_codes)]
    shp_nuts = shp_nuts.to_crs(crs='epsg:4326')

    shp_gdf = shp_nuts.copy()

    shp_cntry = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=0")
    shp_cntry = shp_cntry[shp_cntry['CNTR_CODE'].isin(country_codes)]
    shp_cntry = shp_cntry.to_crs(crs='epsg:4326')

    # get list of geometries
    geoms = shp_cntry['geometry'].tolist()

    # iterate over all combinations of polygons and get the intersections (overlaps)
    overlaps = gpd.GeoDataFrame(gpd.GeoSeries([poly[0].intersection(poly[1]) for poly in itertools.combinations(geoms, 2) if poly[0].intersects(poly[1])]), columns=['geometry'])

    # set the crs
    overlaps.crs = shp_cntry.crs
    shp_cntry1 = shp_cntry.overlay(overlaps, how='difference')


    ###grab processed tick data to plot
    #grab processed tick data from local
    path_to_read_file=f'../../data/{modelname}/processed-master-database/'
    file_name_microfoci = 'cluster_df.csv'
    tick_df = pd.read_csv(path_to_read_file+file_name_microfoci)

    gdf = gpd.GeoDataFrame(
        tick_df, geometry=gpd.points_from_xy(tick_df['longitude'], tick_df['latitude']))
    gdf = gdf.drop(['Unnamed: 0','index_right'],axis=1)

    if region == 'n':
        df_foci = tick_df[tick_df['country_code'].isin(country_codes)]
        print('region==False')
    
    else:
        ### need to do the nuts3 filtering below after choropleth
        # keep_nuts3 = list(set(df_foci.NUTS_ID))
        # shp_gdf = shp_gdf[shp_gdf.NUTS_ID.isin(keep_nuts3)]

    
        shp_cntry = gpd.read_file(f"../../data/{modelname}/geographic-clustering/region-shp-polygons/region_{region}.shp")

        shp_cntry1 = shp_cntry.drop([col for col in shp_cntry.columns if col !='geometry'], axis=1)


        df_foci = gpd.sjoin(gdf,shp_cntry1, how='inner',predicate='intersects')
        keep_countries = list(set(df_foci.country_code))

    ### Begin plotting in folium
    # location = [59, 14.5]
    location = [df_foci.latitude.mean()+4.7, df_foci.longitude.mean()-1.8]


    f = folium.Figure(width=width, height=height)
    m = folium.Map(location=location, zoom_start=zoom_level, 
                tiles=None,
                width=width, height=height,
                control_scale=False,
                zoom_control=False,
                min_zoom=2,
                max_zoom=15,
                zoomDelta=0.5,
                zoomSnap=0.5, #Faster debounce time while zooming
                wheelDebounceTime=20,
                wheelPxPerZoomLevel=20
                #    scrollWheelZoom=False, # disable original zoom function
                #    smoothWheelZoom=True,  # enable smooth zoom 
                #    smoothSensitivity=1   # zoom speed. default is 1
                ).add_to(f)
    
    #change bg color to white
    m = bg_changer(m,'white')

        # colors=['#bbface', '#26ff80', '#029c42', 'green', 'yellow', 'orange', 'red'],
        # index=[0,10,20,35,55,60,80],

    linear = branca.colormap.StepColormap(
        colors=['#f4f4f4','#D0F0D4', '#00ab47', 'yellow','orange','red', '#910101'],
        index=[0,5,10,20,40,60,80],
        vmin=0, vmax=100,
        caption='TBEV Foci Probability (%)' #Caption for Color scale or Legend
    )

    ### below is the same code for maxent output
    # linear = branca.colormap.StepColormap(
    #     colors=['#D0F0D4', '#00ab47', 'green', 'yellow','orange','red', '#910101'],
    #     index=[0,10,20,30,45,60,80],
    #     vmin=0, vmax=100,
    #     caption='TBEV Foci Probability (%)' #Caption for Color scale or Legend
    # )
    # linear.add_to(m)

    predictions = folium.FeatureGroup('Heatmap')
    # #heatmap method
    # HeatMap(
    #     test_df[['lat','lon','pred']],
    #     min_opacity=0.3, radius=25, blur=15, overlay=True, control=True, show=True,
    #     gradient={0.7: 'blue', 0.9: 'lime', 0.91: 'yellow', 0.92: 'orange', .94: 'red'}).add_to(predictions)
    # predictions.add_to(m)

    #sampling points method:
    test_sample = df.sample(dot_sample)
    test_sample['proba'] = np.round(test_sample.pred*100)

    test_sample = gpd.GeoDataFrame(
        test_sample, geometry=gpd.points_from_xy(test_sample['longitude'], test_sample['latitude']))
    test_sample = test_sample[['geometry','proba']]

    #get test_sample vals inside valid countries (remove pts outside countries but inside regions)
    test_sample = test_sample.overlay(shp_cntry1, how='intersection')

    for _, r in test_sample.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                zoom_on_click=False,
                                tooltip = f"Probability: {str(r['proba'])}%",
                                smooth_factor=1,
                                marker=folium.Circle(
                                    radius=2300,
                                    fill_color=linear(r['proba']), 
                                    fill_opacity=0.5, 
                                    color=linear(r['proba']), 
                                    weight=1,
                                    opacity = .2
                                ))
        geo_j.add_to(predictions)
    predictions.add_to(m)

    ### plot countries, localities uses choropleth filtering below (Or using region)
    fg_base = folium.FeatureGroup(name='countries')
    for _, r in shp_cntry1.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(.0001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            #tooltip=f"<b>District: {r['NAME_2']}",
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .4,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)


    # Define your HTML content for the legend
    legend_html = '''
    <div style="position: fixed; 
                top: 11px; left: 50px; width: 110px; 
                background-color: white; border:1px solid black; z-index:9999; 
                font-size:14px; display: flex; flex-direction: column; 
                align-items: center;  padding: 5px;">
        <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Relative Suitability </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #910101; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">80%+</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: red; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">60%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: orange; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">40%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: yellow; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">20%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #00ab47; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">10%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #D0F0D4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">5%</span>
        </div>
            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #f4f4f4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">0%</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)

    #folium.LayerControl().add_to(m)
    m.get_root().add_child(legend)


    ### NEXT SECTION IS FOR CHOROPLETH MAP

    ### map NUTS3 localities to prediction points ###
    pred_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude'])).reset_index(drop=True)

    merged_intersection_adm = gpd.sjoin(pred_gdf,shp_gdf,how='left',predicate='intersects')

    #get missed rows
    leftovers = merged_intersection_adm[merged_intersection_adm['NUTS_ID'].isnull()][['latitude','longitude','pred','geometry']]

    #save properly merged rows
    merged_intersection_adm = merged_intersection_adm[~merged_intersection_adm['NUTS_ID'].isnull()]

    #nearest join with leftover points
    merged_closest_adm= gpd.sjoin_nearest(leftovers, shp_gdf, how='left', distance_col='distance_from_boundary')

    #concatenate intersected and nearest dfs
    merged_adm = pd.concat([merged_intersection_adm, merged_closest_adm]).reset_index(drop=True).drop('distance_from_boundary',axis=1)

    #merged_adm = merged_adm[['index_right','longitude','latitude','pred','NUTS_ID','NUTS_NAME','CNTR_CODE','geometry']]
    
    #drop any districts with no predictions
    merged_adm = merged_adm.dropna(subset=['pred'])

    df_map = merged_adm.groupby(['index_right','CNTR_CODE','NUTS_ID'])['pred'].mean().reset_index()
    df_map['Probability'] = df_map.pred*100
    df_map['Probability'] = df_map['Probability'].round(1)
    df_map['unique_id_str'] = df_map.index.astype('str')
    df_map['District_Name'] = df_map['NUTS_ID']# +': ' +df_map['NUTS_NAME']
    gdf_map = shp_gdf.merge(df_map,how='inner',on=['NUTS_ID']).reset_index()
    gdf_map['geometry'] = gdf_map['geometry'].simplify(.001)
    gdf_map = gdf_map.to_crs('epsg:4326')


    gdf_map = gdf_map[['pred','NUTS_ID','NUTS_NAME','CNTR_CODE_x','geometry','District_Name','Probability']]
    gdf_map= gdf_map.rename({'NUTS_NAME':'Nuts3 District'},axis=1)
    #del merged_adm, merged_closest_adm, merged_intersection_adm 



    ### output files to shapefile for further editing
    # gdf_map.to_file('prediction_by_NUTS3')


    ### create choropleth map 'ma'
    #location set in dotmap m above
    #map configs to adjust zoom settings
    fig = folium.Figure(width=width, height=height)
    ma = folium.Map(location=location, zoom_start=zoom_level, tiles=None, width=width, height=height,
                control_scale=False,
                zoom_control=False,
                min_zoom=2,
                max_zoom=15,
                zoomDelta=0.5,
                zoomSnap=0.25, #Faster debounce time while zooming
                wheelDebounceTime=20,
                wheelPxPerZoomLevel=20
                #    scrollWheelZoom=False, # disable original zoom function
                #    smoothWheelZoom=True,  # enable smooth zoom 
                #    smoothSensitivity=1   # zoom speed. default is 1
                ).add_to(fig)

    #change bg color to white
    ma = bg_changer(ma,'white')


#['#FEF0D9', '#FDD49E', '#FDBB84','#FC8D59', '#EF6548', '#D7301F', '#990000']
    binning = [0,5,10,20,40,60,80,100]
    ### Add in custom choropleth legend
    # Define your custom color palette as a LinearColormap
    #https://github.com/python-visualization/folium/issues/403
    custom_palette = cmp.StepColormap(
        ['#f4f4f4','#D0F0D4', '#00ab47', 'yellow','orange','red', '#910101'],
        index=binning,
        vmin=0,
        vmax=100,
        caption='TBEV Foci Relative Probability (%)'
    )

    #create choropleth map
    cp = folium.Choropleth(
        geo_data = gdf_map.to_json(drop_id=True),
        name = 'choropleth',
        data = gdf_map,
        columns = ("District_Name","Probability"),
        key_on = "feature.properties.District_Name",
        bins=binning,
        nan_fill_color = 'grey',
        line_opacity = 1,
        legend_name = 'Foci Relative Probability (%)'
    )

    ### delete colormap legends
    for key in cp._children:
        if key.startswith('color_map'):
            print(key)
            del cp._children[key]

    # creating a state indexed version of the dataframe so we can lookup values
    data_indexed = gdf_map.set_index('Nuts3 District')

    df_dict = gdf_map.set_index('District_Name')['Probability']
    style_function = lambda x: {'fillColor': custom_palette(df_dict[x['properties']['District_Name']]),
                                'color': 'black',  # Line color
                                'weight': 0.1,  # Line width
                                'fillOpacity': 0.75
        }
    
    cp.geojson.style_function = style_function
    cp.add_to(ma)

    # looping thru the geojson object and adding a new property
    # and assigning a value from our dataframe

    for idx,s in enumerate(cp.geojson.data['features']):
        s['properties']['Probability1'] = str(data_indexed.loc[s['properties']['Nuts3 District'], 'Probability'])+'%'

    ##and finally adding a tooltip/hover to the choropleth's geojson
    folium.GeoJsonTooltip(['Nuts3 District', 'Probability1']).add_to(cp.geojson)


    #['#D0F0D4', '#00ab47', 'yellow','orange','red', '#910101']`
    # Define your HTML content for the legend
    legend_html = '''
    <div style="position: fixed; 
                top: 11px; left: 50px; width: 110px; 
                background-color: white; border:1px solid black; z-index:9999; 
                font-size:14px; display: flex; flex-direction: column; 
                align-items: center;  padding: 5px;">
        <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Relative Suitability </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #910101; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">80%+</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: red; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">60%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: orange; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">40%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: yellow; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">20%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #00ab47; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">10%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #D0F0D4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">5%</span>
        </div>
        <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
            <div style="background-color: #f4f4f4; width: 20px; height: 20px;"></div>
            <span style="font-size:10px; margin-left: 2px;">0%</span>
        </div>
    </div>
    '''
    ma.get_root().html.add_child(folium.Element(legend_html))

    ### USE CHOROPLETH FILTERING FOR NUTS3 for map m (dotmap)

    ### plot localities
    fg_base = folium.FeatureGroup(name='NUTS3 Districts')
    for _, r in gdf_map.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'].simplify(.001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .1,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)

    ### plot countries, localities uses choropleth filtering below (Or using region)
    fg_base = folium.FeatureGroup(name='countries')
    for _, r in shp_cntry1.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(.0001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            #tooltip=f"<b>District: {r['NAME_2']}",
                            style_function=lambda x: {
                                    "fillColor": "orange",
                                    "fillOpacity": 0,
                                    "weight": .4,
                                    "color":  "black",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(ma)


    return m, ma





def plot_covariates(df, cov, dot_sample=50000, colorscale = ['white','darkblue'], hide_legend=False):
    #Takes in dataframe with columns ['lat_env','lon_env','cov'] and outputs predictor map
    width = 800
    height = 800

    location = [df.lat_env.mean()+2.2, df.lon_env.mean()+5]
    country_codes =  ['DK','NO','SE','FI','AT','CH','CZ','DE','EE','FR','LT','LV','NL','PL','SK', 'IT','UK','HR','BE','SI','LU']
    anti_country_codes = ['TR','EL','BG','MK','AL','RO','ME','RS','IS','CY']
    

    # Must run this for ANY map production, even choropleth maps!!
    ctry_map = path.ctry_map

    ## all localities in one gdf
    shp_nuts = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=3")
    shp_nuts = shp_nuts[shp_nuts['CNTR_CODE'].isin(country_codes)]
    shp_nuts = shp_nuts.to_crs(crs='epsg:4326')

    shp_gdf = shp_nuts.copy()

    shp_cntry = gpd.read_file(f"../../data/raw-data/shapefiles/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp", 
                    where=f"LEVL_CODE=0")
    shp_cntry = shp_cntry[shp_cntry['CNTR_CODE'].isin(country_codes)]
    shp_cntry = shp_cntry.to_crs(crs='epsg:4326')

    # get list of geometries
    geoms = shp_cntry['geometry'].tolist()

    # iterate over all combinations of polygons and get the intersections (overlaps)
    overlaps = gpd.GeoDataFrame(gpd.GeoSeries([poly[0].intersection(poly[1]) for poly in itertools.combinations(geoms, 2) if poly[0].intersects(poly[1])]), columns=['geometry'])

    # set the crs
    overlaps.crs = shp_cntry.crs
    shp_cntry1 = shp_cntry.overlay(overlaps, how='difference')

    attr = (
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
        'contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    )
    tiles = 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png'


    f = folium.Figure(width=width, height=height)
    m = folium.Map(location=location, zoom_start=4, 
                tiles=None,
                width=width, height=height,
                zoom_control=False,
                scrollWheelZoom=False,
                dragging=False
                ).add_to(f)

    ###add css to change bg color to white
    map_id = m._id
    print(map_id)
    bg_css = "<style>#map_" + str(map_id) + """
        {
            background-color: white; /* Set background to white */
        }
    </style>
    """
    Element(bg_css).add_to(m.get_root().header)

    ### plot dotmaps of covariates and define linearcolormap
    linear = branca.colormap.LinearColormap(
        colors=colorscale,
        index=[df[cov].min(), df[cov].max()],
        vmin=df[cov].min(),
        vmax=df[cov].max()
    )

    predictions = folium.FeatureGroup('Heatmap')
    linear.add_to(m)

    #sampling points method:
    test_sample = df.sample(dot_sample)

    test_sample = gpd.GeoDataFrame(
        test_sample, geometry=gpd.points_from_xy(test_sample['lon_env'], test_sample['lat_env']))
    test_sample = test_sample[['geometry',cov]]

    # get test_sample vals inside valid countries
    test_sample = test_sample.overlay(shp_cntry, how='intersection')

    for _, r in test_sample.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                                zoom_on_click=False,
                                #tooltip = f"Value: {str(r[cov])}",
                                smooth_factor=1,
                                marker=folium.Circle(
                                    radius=2500,
                                    fill_color=linear(r[cov]), 
                                    fill_opacity=0.3, 
                                    color=linear(r[cov]), 
                                    weight=1,
                                    opacity = .2
                                ))
        geo_j.add_to(predictions)
    predictions.add_to(m)

    def region_style(feature):
        return {"fillColor": feature['properties']['color'],
                "fillOpacity": .2,
                "weight": .3,
                "color":  "black",
                'interactive':False}
    
        ### plot countries, localities uses choropleth filtering below (Or using region)
    fg_base = folium.FeatureGroup(name='countries')
    for _, r in shp_cntry1.iterrows():
        sim_geo = gpd.GeoSeries(r['geometry'])#.simplify(.0001))
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j,
                            #tooltip=f"<b>District: {r['NAME_2']}",
                            style_function=lambda x: {
                                    "fillColor": "#FBFCFC",
                                    "fillOpacity": 0,
                                    "weight": .5,
                                    "color":  "black",
                                    #'opacity': ".7",
                                    'interactive':False}
            )
        geo_j.add_to(fg_base)
    fg_base.add_to(m)


    # #['#D0F0D4','#fde725','#5ec962','#21918c','#3b528b','#440154']
    # # Define your HTML content for the legend
    # legend_html = '''
    # <div style="position: fixed; 
    #             top: 11px; left: 50px; width: 100px; 
    #             background-color: white; border:1px solid black; z-index:9999; 
    #             font-size:14px; display: flex; flex-direction: column; 
    #             align-items: center;  padding: 5px;">
    #     <div style="text-align: center; width: 100%; margin-bottom: 5px; font-weight: bold; font-size:10px;"> Annual Wet Days (# days) </div>
    #         <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
    #         <div style="background-color: #910101; width: 20px; height: 20px;"></div>
    #         <span style="font-size:10px; margin-left: 2px;">80%+</span>
    #     </div>
    #     <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
    #         <div style="background-color: red; width: 20px; height: 20px;"></div>
    #         <span style="font-size:10px; margin-left: 2px;">60%</span>
    #     </div>
    #     <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
    #         <div style="background-color: orange; width: 20px; height: 20px;"></div>
    #         <span style="font-size:10px; margin-left: 2px;">40%</span>
    #     </div>
    #     <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
    #         <div style="background-color: yellow; width: 20px; height: 20px;"></div>
    #         <span style="font-size:10px; margin-left: 2px;">20%</span>
    #     </div>
    #     <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
    #         <div style="background-color: #00ab47; width: 20px; height: 20px;"></div>
    #         <span style="font-size:10px; margin-left: 2px;">117</span>
    #     </div>
    #     <div style="display: flex; width: 100%; align-items: center; margin-bottom: 1px;">
    #         <div style="background-color: #D0F0D4; width: 20px; height: 20px;"></div>
    #         <span style="font-size:10px; margin-left: 2px;">38</span>
    #     </div>
    #     </div>
    # </div>
    # '''
    # m.get_root().html.add_child(folium.Element(legend_html))


    # legend = branca.element.MacroElement()
    # legend._template = branca.element.Template(legend_html)

    if hide_legend==True:
        drop_keys=[]
        for key in m._children:
            if key.startswith('color_map'):
                drop_keys.append(key)
                print(key)
        del m._children[drop_keys[0]]

    return m

