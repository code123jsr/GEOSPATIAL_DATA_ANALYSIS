import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import datetime as dt
import folium
import googlemaps
import glob
import os
import branca.colormap as cm
import vincent
import json

def read_dataset():
    '''
    Combines all of the file data and lists the amount of citations per day in order
    :return: DataFrame of our dataset
    '''

    read_list_rev = ['parking_citations_2012_part1_datasd.csv', 'parking_citations_2012_part2_datasd.csv',
                'parking_citations_2013_part1_datasd.csv', 'parking_citations_2013_part2_datasd.csv',
                'parking_citations_2014_part1_datasd.csv', 'parking_citations_2014_part2_datasd.csv',
                'parking_citations_2015_part1_datasd.csv', 'parking_citations_2015_part2_datasd.csv',
                'parking_citations_2016_part1_datasd.csv', 'parking_citations_2016_part2_datasd.csv',
                'parking_citations_2017_part1_datasd.csv', 'parking_citations_2017_part2_datasd.csv',
                'parking_citations_2018_part1_datasd.csv', 'parking_citations_2018_part2_datasd.csv',
                ]
    read_list_norm = ['parking_citations_2019_part1_datasd.csv', 'parking_citations_2019_part2_datasd.csv',
                'parking_citations_2020_part1_datasd.csv', 'parking_citations_2020_part2_datasd.csv',
                'parking_citations_2021_part1_datasd.csv', 'parking_citations_2021_part2_datasd.csv',
                'parking_citations_2022_part1_datasd.csv', 'parking_citations_2022_part2_datasd.csv',
                'parking_citations_2023_part1_datasd.csv', 'parking_citations_2023_part2_datasd.csv'
                ]
    
    con_list = []
    for f in read_list_rev:
        df = pd.read_csv('./Datasets/datasets/' + f) # make sure to change this to the correct python read path
        df = df.reindex(index=df.index[::-1])
        con_list.append(df)
    for f in read_list_norm:
        df = pd.read_csv('./Datasets/datasets/' + f) # make sure to change this to the correct python read path
        con_list.append(df)
    
    output_df = pd.concat(con_list)
    return output_df

def read_geo_data(dir_path, start_year):
    """
    This function reads in the dataset csv files, for geo-coordinates creation.
    Here, we read dataset files from a specific start year which user can specify, 
    since the density of places crashes the folium rendering.
    Inputs:
    dirpath: Path of input CSV files.
    start_year: From which year the files needs to be read.
    Output: merged geopandas dataframe.
    """
    assert (isinstance(start_year, int)) #Year should be specified as a Integer
    assert (isinstance(dir_path, str)) # Directory Path Should be String.
    years_range = np.arange(start_year,start_year+2).tolist()
    input_geo_files = glob.glob(dir_path)
    geo_dataframes = list()
    for dataset_file in input_geo_files:
        dataset_year=int(os.path.basename(dataset_file).split("_")[2])
        if dataset_year in years_range:
            geo_dataframes.append(pd.read_csv(dataset_file))
    merged_geo_dataframe = pd.concat(geo_dataframes, ignore_index=True)
    assert(isinstance(merged_geo_dataframe, pd.DataFrame))
    return merged_geo_dataframe     

def get_geo_coordinates(geo_dataframe, gmaps, min_citations):
    """
    This function uses the Google Maps Python Module to get the latitude, longitude of a street address which we 
    got from the database and returns that to a dictionary.
    Inputs: 
    geo_dataframe: pandas dataframe created from dataset for a number of years specified by the user.
    gmaps: object from google maps python module.
    min_citations: to extract a location, how many minimum number of citations should be there.
    Output:
    citations_geo_dict: dictionary contatining location coords of the street addresses having minimum number of citations.
    """
    assert(isinstance(geo_dataframe, pd.DataFrame)) # Should be a DataFrame
    assert(isinstance(min_citations, float)) #Should be an integer.
    citations_geo_dict = dict()
    citations_count_list = geo_dataframe['location'].value_counts().tolist()
    for idx, name in enumerate(geo_dataframe['location'].value_counts().index.tolist()):
        citations_count = citations_count_list[idx]
        if citations_count > min_citations:
            citations_geo_dict[name] = {}
            citations_geo_dict[name]['location'] = [gmaps.geocode(f'{name}, San Diego, CA')[0]['geometry']['location']['lat'], gmaps.geocode(f'{name}, San Diego, CA')[0]['geometry']['location']['lng']]
    assert(isinstance(citations_geo_dict, dict))
    return citations_geo_dict

def create_date_month_df(df):
    """
    Since, we need to plot the database based on monthly and yearly citations we need to create the datetime object from the column
    and create month and year column in the dataset. This function essentially accomplishes this.
    Input: 
    df: The full dataset read in the beginning.
    df: returns the updated database with month and year column.
    """
    assert(isinstance(df, pd.DataFrame))
    df['date_issue'] = pd.to_datetime(df['date_issue']) 
    df['year'] = df['date_issue'].dt.year
    df['month'] = df['date_issue'].dt.month
    assert(isinstance(df, pd.DataFrame))
    return df

def get_citations_count(df, min_citations):
    """
    The function takes in the dataset and for the full dataset returns a dict, which has all the places of citations as keys and number of overall violations as value.
    and since the dataset is huge, taking only thr places which have more than a certain min_citations.
    Input: The full dataset read in the beginning.
    Output: Dict with all the places of citations and their count.
    """
    assert(isinstance(df, pd.DataFrame))
    assert(isinstance(min_citations, float))
    citations_count_dict = {}
    citations_count_list = list()
    for location, vio_count in df['location'].value_counts().items():
        if float(vio_count) > min_citations:
            citations_count_dict[location] = vio_count
            citations_count_list.append(float(vio_count))
    return citations_count_dict, citations_count_list

def create_geo_dataframe_dict(df, citations_geo_dict):
    """
    The function gets the citations_geo_dict which has the filtered citation loactions and their coords.
    It will append the monthly, yearly ditribution for that location from the dataset and appends those keys to the 
    location dict.
    Input: 
    df: the dataset dataframe
    citations_geo_dict: dictionary of all the citations locations with their geo coords.
    Output: 
    citations_geo_dict: dictionary of all the citations locations with their 
                        geo coords
                        monthly dist.
                        yearly dist.
                        top 3 vios.
                        average fine
    """
    assert(isinstance(df, pd.DataFrame))
    assert(isinstance(citations_geo_dict, dict))
    for street_address in citations_geo_dict:
        # Required Few default variables.
        # Gregorian Calendar Years as they are used after 1582.
        year_dict = {2012:0, 2013:0, 2014:0, 2015:0, 2016:0, 2017:0, 2018:0, 2019:0, 2020:0, 2021:0, 2022:0, 2023:0} 
        # Months Inside that Calendar, to represent the parking tickets.
        month_dict = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
        location_df = df[df['location'] == street_address]
        citations_geo_dict[street_address]['yearly'] = {}
        for idx, name in location_df['year'].value_counts().items():
            year_dict[idx] = name
        citations_geo_dict[street_address]['yearly'] = year_dict
    
        #Next we do monthly:
        citations_geo_dict[street_address]['monthly'] = {}
        for idx, name in location_df['month'].value_counts().items():
            month_dict[idx] = name
        citations_geo_dict[street_address]['monthly'] = month_dict
        
        # Next We do for type of Violations.
        citations_geo_dict[street_address]['vios_type'] = {}
        vios_dict = {}
        for idx, name in location_df['vio_desc'].value_counts().nlargest(10).items():
            vios_dict[idx] = name
        citations_geo_dict[street_address]['vios_type'] = vios_dict
    
        #Calculating the average fine.
        citations_geo_dict[street_address]['Average_fine'] = round(location_df['vio_fine'].mean(axis=0),3)   
    return citations_geo_dict

def create_map_markers(distribution, citations_geo_dict, plotting_list, legend_string, citations_count_list, citations_count_dict, m):
    """
    This function adds the markers over the folium map where the citations were given and from the full dataset it takes in geo coords, monthly dist., yearly dist., top 3 vios.,
    average fine distributions and add them as Vega Grpahs over the folium Markers.
    These folium markers are subsequently added to the folium map which was global variable.
    Inputs:
    distribution: string whether is monthly yearly or vios
    citations_geo_dict: dictionary of all the citations locations with their 
                        geo coords
                        monthly dist.
                        yearly dist.
                        top 3 vios.
                        average fine
    plotting_list: the x legend axis naming of the Vega Map.
    legend_string: the x legend naming of the Vega Map.
    citations_count_list: the number of citations for all the locations.    
    """
    assert(isinstance(citations_geo_dict, dict))
    assert(isinstance(distribution, str))
    assert(isinstance(plotting_list, list))
    assert(isinstance(legend_string, str)) 
    assert(isinstance(citations_count_list, list))
    gregory_months_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    linear = cm.LinearColormap(["green", "yellow", "orange", "red"], caption='Parking Violations Count', vmin=100, vmax=max(citations_count_list), index=[500, 1000, 2500, 5000], tick_labels=[500, 1500, 3500, 10000])
    feature_group = folium.FeatureGroup(name=f"{distribution} Citations", control=True)
    for street_address in citations_geo_dict:
        if distribution=='monthly':plotting_list = gregory_months_list
        else: plotting_list= list(citations_geo_dict[street_address][distribution].keys())       
        plot_dict = {f'{legend_string}': plotting_list, 'data': list(citations_geo_dict[street_address][f'{distribution}'].values())}
        marker_chart = vincent.Bar(plot_dict, iter_idx=f'{legend_string}', height=200, width=600)
        marker_chart.axis_titles(x=f'{legend_string} ({street_address})', y='No. of Citations')
        plot_data = json.loads(marker_chart.to_json())
        tooltip_text = f"{street_address} | Average_Fine: {citations_geo_dict[street_address]['Average_fine']}$"
        circle_marker = folium.CircleMarker(
                            location=citations_geo_dict[street_address]['location'],
                            radius=10,
                            color=linear(citations_count_dict[street_address]),
                            stroke=False,
                            fill=True,
                            fill_opacity=0.6,
                            opacity=1,
                            tooltip='<h6><b>{}</b></h6>'.format(tooltip_text)).add_to(feature_group)
        popup = folium.Popup(street_address).add_to(circle_marker)
        folium.Vega(plot_data, width="100%", height="100%").add_to(popup)
    feature_group.add_to(m)
    return m



def create_map_markers_vios(distribution, citations_geo_dict, plotting_list, legend_string, citations_count_list, citations_count_dict, m):
    """
    This function adds the markers only for vios distribution over the folium map where the citations were given and from the full dataset it takes in geo coords, monthly dist., yearly dist., top 3 vios.,
    average fine distributions and add them as Vega Grpahs over the folium Markers.
    These folium markers are subsequently added to the folium map which was global variable.
    Inputs:
    distribution: string whether is monthly yearly or vios
    citations_geo_dict: dictionary of all the citations locations with their 
                        geo coords
                        monthly dist.
                        yearly dist.
                        top 3 vios.
                        average fine
    plotting_list: the x legend axis naming of the Vega Map.
    legend_string: the x legend naming of the Vega Map.
    citations_count_list: the number of citations for all the locations.    
    """
    assert(isinstance(citations_geo_dict, dict))
    assert(isinstance(distribution, str))
    assert(isinstance(plotting_list, list))
    assert(isinstance(legend_string, str)) 
    assert(isinstance(citations_count_list, list))
    gregory_months_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    linear = cm.LinearColormap(["green", "yellow", "orange", "red"], caption='Parking Violations Count', vmin=100, vmax=max(citations_count_list), index=[500, 1000, 2500, 5000], tick_labels=[500, 1500, 3500, 10000])
    vios_feature_group = folium.FeatureGroup(name=f'{distribution}', control=True)
    for street_address in citations_geo_dict:
        vios_type = list(citations_geo_dict[street_address][f'{legend_string}'].keys())
        vios_type_values = list(citations_geo_dict[street_address][f'{legend_string}'].values())
        plot_dict = {f'{legend_string}': vios_type[:3] , 'data': vios_type_values[:3]}
        marker_vios_chart = vincent.Bar(plot_dict, iter_idx=f'{legend_string}', height=200, width=800)
        marker_vios_chart.axis_titles(x=f'Citations Reasons ({street_address})', y='Citations Count')
        plot_vios_data = json.loads(marker_vios_chart.to_json())
        tooltip_text = f"{street_address} | Average_Fine: {citations_geo_dict[street_address]['Average_fine']}$"
        vios_circle_marker = folium.CircleMarker(
                            location=citations_geo_dict[street_address]['location'],
                            radius=10,
                            color=linear(citations_count_dict[street_address]),
                            stroke=False,
                            fill=True,
                            fill_opacity=0.6,
                            opacity=1,
                            tooltip='<h6><b>{}</b></h6>'.format(tooltip_text)).add_to(vios_feature_group)
        vios_popup = folium.Popup(street_address).add_to(vios_circle_marker)
        folium.Vega(plot_vios_data, width="100%", height="100%").add_to(vios_popup)
    vios_feature_group.add_to(m)
    return m


def create_geo_spatial_analysis(df, gmaps):
    """
    This function calls the above functions and collects all the collateral and finally returns the folium Map:
    Input:
    df: the dataset dataframe
    gmaps: google maps object with user client id
    Outputs:
    folium Map: the map reperesentation with all the citation locations and their distributions.    
    """
    # Gregorian Months as a GLobal Variable.
    #Folium Map as a Global Variable.
    m = folium.Map(location=(32.7157, -117.1611), tiles = 'OpenStreetMap', zoom_start=12)
    gregory_months_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    geo_dataframe = read_geo_data('./Datasets/datasets/*csv', 2022)
    citations_geo_dict = get_geo_coordinates(geo_dataframe, gmaps, 100.0)
    df = create_date_month_df(df)
    citations_count_dict, citations_count_list = get_citations_count(df, 100.0)
    citations_geo_dict = create_geo_dataframe_dict(df, citations_geo_dict)
    folium_map = create_map_markers('monthly', citations_geo_dict, gregory_months_list, 'months', citations_count_list, citations_count_dict, m)
    folium_map = create_map_markers('yearly', citations_geo_dict, [], 'years', citations_count_list, citations_count_dict, folium_map)
    folium_map = create_map_markers_vios('Top 3 Citations', citations_geo_dict, [], 'vios_type', citations_count_list, citations_count_dict, folium_map)
    return folium_map




if __name__ == "__main__":
    df = read_dataset()
    gmaps = googlemaps.Client(key='')
    folium_map = create_geo_spatial_analysis(df, gmaps)
    folium.LayerControl(collapsed=False).add_to(folium_map)
