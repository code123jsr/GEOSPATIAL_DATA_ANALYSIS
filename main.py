# main.py

import pandas as pd
import time_series_analysis
import EDA
import geo_spatial_analysis
import googlemaps

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
        df = pd.read_csv('./Datasets/Parking Ticket Databases' + f) # make sure to change this to the correct python read path
        df = df.reindex(index=df.index[::-1])
        con_list.append(df)
    for f in read_list_norm:
        df = pd.read_csv('./Datasets/Parking Ticket Databases' + f) # make sure to change this to the correct python read path
        con_list.append(df)
    
    output_df = pd.concat(con_list)
    return output_df


if __name__ == "__main__":

    df = read_dataset()
    
    # exploratory data analysis
    
    # getting the top10 locations
    location_count=EDA.locations_by_count(df)
    
    # plotting EDA distributions
    EDA.top_location_monthly_count(df, 'SKI BEACH LOT 3000 INGRAHAM ST')
    EDA.top_location_monthly_count(df, '3600 5TH AV')
    EDA.plot_citations_per_year(df, '3600 5TH AV')
    EDA.top_locations(df)
    EDA.table_counts(df)
    EDA.fine_by_description(df)
    CitatationDate=EDA.date_by_citation(df)

    # population regression analysis (Exclude 2020)
    population_p_value = time_series_analysis.population_correlation(df, include_2020=False, visualization=True)

    # parking meters analysis
    parking_meters_p_value = time_series_analysis.parking_meters_correlation(df, visualization = True)

    # existing parking meters and parking ticket frequency changes scatter graph
    time_series_analysis.parking_meters_scatter_plot(df)

    # heatmap for October, November. December and for monthly of past decade
    time_series_analysis.time_heatmap_monthly(df, month = 10)
    time_series_analysis.time_heatmap_monthly(df, month = 11)
    time_series_analysis.time_heatmap_monthly(df,month = 12)
    time_series_analysis.time_heatmap_yearly(df)

    # seasonal decompose
    seasonal_decomposition = time_series_analysis.seasonal_decompose(df, visualization=True)
    
    # geospatial analysis
    gmaps = googlemaps.Client(key='')
    folium_map = geo_spatial_analysis.create_geo_spatial_analysis(df, gmaps)
    folium.LayerControl(collapsed=False).add_to(folium_map)
        
