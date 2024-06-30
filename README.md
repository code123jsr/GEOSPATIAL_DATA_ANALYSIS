# A San Diego Parking Tickets Data Analysis and GeoSpatial Visualization

With the dataset of Parking Citations in the San Diego area published by the City Treasurer, we conduct a series of analysis including Linear Regression, Geospatial Heat Map Visualization to explore the characteristics of the parking tickets from the perspectives of time and space.  
Our research conclusion shows and explains why there are times and places where more parking tickets can be found, so that we provide some tips for parking when it's almost impossible to park in the parking lot. Moreover, we develop a small client-side web application that will tell a user if they are prone to parking tickets based on the street name and the time, with the help of Google Map API.

Datasets
1. Parking Tickets in San Diego Between 2012-2023
2. Population in Cities of San Diego Between 2012-2022
3. Parking Meters in San Diego Between 2018-2023
4. Parking Lots in San Diego Between 2016-2023


# File Structure

- `Datasets/`: This folder contains all the csv files of Datasets
    - `Parking Ticket Databases/`: original datasets of Parking Tickets in San Diego Between 2012-2023
    - `population_cities_2010_2022.csv`: original dataset of Population in Cities of San Diego Between 2012-2022
    - `parking_meters_loc_datasd.csv`: original dataset of Parking Meters in San Diego Between 2018-2023
    - `park_lots_loc_datasd.csv`: original dataset of Parking Lots in San Diego Between 2016-2023

- `Web App/`: This folder contains files for Website Application
    - `build/`: files for website development
    - `future_scope.py`: codes for parking suggestion generation 
    - `suggestion_model.pkl`: model for suggestion generation
    - `OneHotEncoders.pkl`: one-hot encoders for suggestion generation
 
- `ECE143-Group10_Final Presentation.pdf`: final presentation slides for our project
- `time_series_analysis.py`: functions used for time series analysis including correlation exploration, regression, seasonal decompose, heatmap
- `geo_spatial_analysis.py`: functions used for spatial analysis including interactive maps
- `EDA.py`: It contains the code with functions of various plots like bar graph, pie chart for the analysis of parameters in the dataset
- `Visualization.ipynb`: visualizations of the analysis
- `geospatial_analysis_parking.html`: the interactive map generated from `geo_spatial_analysis.py`
- `main.py`: the main file with all executing codes for our functions
- `requirements.txt`: the required third-party libraries and corresponding versions

# Time_Series_Analysis
Visualizes the number of parking citations using heatmaps over the course of 2012 - 2023. Creates plots of all months, all days in October, November, and December. Creates time series analyses and describes daily, weekly, and seasonal trends

- `Functions/`
    - `time_series_analysis.main()/`:
        - `Desc.`: import datasets as DataFrame
    - `time_series_analysis.date_reframe(df, date_column, daily = False, monthly = False, yearly = False)/`:
        - `df`: original dataframe of databases
        - `date_column`: the column name of date
        - `daily`: if need daily count (bool)
        - `monthly`: if need monthly count (bool)
        - `yearly`: if need yearly count (bool)
        - `Desc.`: the periodic frequency of the tickets
    - `time_series_analysis.regression_visualization(model, x, y, category)/`:
        - `model`: the regression model
        - `x`: the original x variable
        - `y`: the original y variable
        - `category`: the required plot (population/parking meters)
        - `Desc.`: scattered points and linear ploting
    - `time_series_analysis.population_correlation(df, include_2020 = True, visualization = False)/`:
        - `df`: the original DataFrame
        - `include_2020`: if include the impact of pandamic (bool)
        - `visualization`: if show the visualization (bool)
        - `Desc.`: return P-Value of the regression for population and parking tickets correlation analysis
    - `time_series_analysis.parking_meters_correlation(df, visualization = False)/`:
        - `df`: the original DataFrame
        - `visualization`: if show the visualization (bool)
        - `Desc.`: return P-Value of the regression for parking meters and parking tickets correlation analysis
    - `time_series_analysis.parking_meters_scatter_plot(df)/`:
        - `df`: the original DataFrame
        - `Desc.`: for visualization of scattered points of parking meters and tickets
    - `time_series_analysis.lockdown(ds)/`:
        - `ds`: the date 
        - `Desc.`: return result (bool) for identification if the day is during lockdown
    - `time_series_analysis.seasonal_decompose(df, visualization = False)/`:
        - `df`: the original DataFrame 
        - `visualization`: if show the visualization (bool)
        - `Desc.`: for season decomposition analysis
    - `time_series_analysis.time_heatmap_yearly(df)/`:
        - `df`: the original DataFrame 
        - `Desc.`: plot heatmap for months
    - `time_series_analysis.time_heatmap_monthly(df, month)/`:
        - `df`: the original DataFrame 
        - `month`: month for heatmap
        - `Desc.`: plot heatmap for certain month
    - `time_series_analysis.parse_cols_month(df)`
        - `df`: sorted dataframe with parking citation data
        - `Desc.`: combines all of the month's parking citations, returns a dataframe object
    - `time_series_analysis.parse_cols_year(df)`
        - `df`: output dataframe of parse_cols_month(df)
        - `Desc.`: lists the parking citation count by month (rows) and by year (columns), returns a dataframe object
    - `time_series_analysis.parse_cols_daily(month, df)`
        - `df`: sorted dataframe with parking citation data
        - `month`: month to capture data from
        - `Desc.`: combines all of the month's parking citations, returns a dataframe object
    - `time_series_analysis.parse_cols_daily2(df)`
        - `df`: output dataframe of parse_cols_daily(df)
        - `Desc.`: lists the parking citation count by day (rows) and by year (columns), returns a dataframe object
    - `time_series_analysis.parse_cols_basic()`
        - `Desc.`: Combines all of the file data and lists the amount of citations per day in order. Make sure to change the read paths for the .csv files 
    - `future_scope.if_holiday(date)`
        - `date`: string of a date
        - `Desc.`: return bool to identify if the date is a holiday
    - `future_scope.get_nearby_points_of_interest(lat, lng)`
        - `lat`: latitude
        - `lng`: longitude
        - `Desc.`: return bool to identify if the location is near an interest
    - `future_scope.get_address_details_geopy(address)`
        - `address`: string of address within San Diego
        - `Desc.`: return detailed components of address
    - `future_scope.get_nearby_parking(lat, lng)`
        - `lat`: latitude
        - `lng`: longitude
        - `Desc.`: return bool to identify if the location is near a parking meter
    - `future_scope.parking_suggestion(date, location)`
        - `date`: string of query date
        - `location`: string of query location
        - `Desc.`: return bool to identify the risk of parking

    # Exploratory Data Analysis
It represents the visualization of parking citations datasets using bar plots, pie chart, table observing the correlation between parameters like number of citations, location, type of violation, date etc for the period of last 10 years

- `Functions/`
    - `EDA.top_locations(df)`:
        - `df`: original dataframe of databases
        - `Desc.`: returns a pie chart for the top ten locations with the most citations
    - `EDA.locations_by_count(df)`:
        - `df`: original dataframe of databases
        - `Desc`: returns the bar plot for the top 20 locations of citation counts.
    - `EDA.fine_by_description(df)`:
        - `df`: original dataframe of databases
        - `Desc`: returns a bar plot of mean violation fine by violation description
    - `EDA.date_by_citation(df)`:
        - `df`: original dataframe of databases
        - `Desc`: returns the pie chart of dates with top ten citations count .
    - `EDA.top_location_monthly_count(df, 'SKI BEACH LOT 3000 INGRAHAM ST')`:
        - `df`: original dataframe of databases
        - 'SKI BEACH LOT 3000 INGRAHAM ST': location required
        - `Desc`:  returns bar plot for number of citations aggrevated for all months for the highest location 
    - `EDA.top_location_monthly_count(df, '3600 5TH AV')`:
        - `df`: original dataframe of databases
        - '3600 5TH AV': location required
        - `Desc`: returns the bar plot for number of citations aggrevated for all months for the second highest location 
    - `EDA.table_counts(df)`:
        - `df`: original dataframe of databases
        - `Desc`: displays a table containing violation counts for violation description
    - `EDA.plot_citations_per_year(df, '3600 5TH AV')`:
        - `df`: original dataframe of databases
        - '3600 5TH AV': location required for yearwise analysis
        - `Desc`: returns the plot for the number of citations per year for a specific location

# GeoSpatial Analysis
It represents the visualization of parking citations datasets onto the Folium Open StreetMap, the color of the citation loaction shows the intensity of citations.
Hovering over a location shows the name & average fine for last 10 years at the street. And we have yearly, monthly and Top 3 Violations Reasons distribution for every location on the map. 

Since, the Map Visualization is not Available in Visualizations.ipynb due to Jupyter Trust Notebook. A seperate standalone HTML geospatal_analaysis_parking.html is added

- `Functions/`
    - `geo_spatial_analysis.read_geo_data(dirPath, year)`:
        - `dirPath`: location of all dataset csv are present
        - `year`: from which year the dataset needs to be considered
        - `Desc.`: reads in the dataset csv files, for geo-coordinates creation from a specific year and return the dataframe.
    - `geo_spatial_analysis.get_geo_coordinates(geo_dataframe, gmaps, min_citations)`:
        - `geo_dataframe`: pandas dataframe for years specified by the user.
        - `gmaps`: object from google maps python module, based on user Client ID
        - `min_citations`: minimum number of citations to be considered for geo loaction 
        - `Desc.`: the Google Maps Python Module to get the latitude, longitude of a street address which we got from the database and returns that to a dictionary.
    - `geo_spatial_analysis.create_date_month_df(df)`:
        - `df`: the full dataset pandas dataframe
        - `Desc.`: create month and year column in the dataset
    - `geo_spatial_analysis.get_citations_count(df, min_citations)`:
        - `df`: the full dataset pandas dataframe
        - `min_citations`: minimum number of citations to consider
        - `Desc.`: function takes in the dataset and for the full dataset returns a dict
    - `geo_spatial_analysis.create_geo_dataframe_dict(df, citations_geo_dict)`:
        - `df`: the full dataset pandas dataframe
        - `citations_geo_dict`: dictionary of all the citations locations
        - `Desc.`: append the monthly, yearly ditribution for that location from the dataset and appends those keys to the 
    location dict.

    - `geo_spatial_analysis.create_map_markers(distribution, citations_geo_dict, plotting_list, legend_string, citations_count_list, m)`:
        - `distribution`: string whether is monthly yearly or vios
        - `citations_geo_dict`: dictionary of all the citations locations
        - `plotting_list`: dictionary of all the citations loctions
        - `legend_string`: dictionary of all the citations loctions
        - `citations_count_list`: dictionary of all the citations loctions
        - `m`: dictionary of all the citations loctions
        - `Desc.`: adds the markers over the folium map
    - `geo_spatial_analysis.create_map_marker_vios(distribution, citations_geo_dict, plotting_list, legend_string, citations_count_list, m)`:
        - `distribution`: string whether is monthly yearly or vios
        - `citations_geo_dict`: dictionary of all the citations locations
        - `plotting_list`: dictionary of all the citations loctions
        - `legend_string`: dictionary of all the citations loctions
        - `citations_count_list`: dictionary of all the citations loctions
        - `m`: dictionary of all the citations loctions
        - `Desc.`: adds the markers over the folium map 
                 

# Usage

To display time series graphics

1. 'import time_series_analysis'

2. ensure that the dataframe with parking citation data has been created and is in order of time

3. call 'time_series_analysis.main(df)', where df is the dataframe consisting of parking citation data

end

To get parking suggestion

1. 'import future_scope'

2. call 'future_scope.parking_suggestion(date, location)', where date and location are the query date and location

end

To display the data analysis plot

1. 'import EDA'

2. Check for the dataframe with parking citation data if it has been created

3. call 'EDA.main(df)', where a is the dataframe consisting of parking citation data

end

To display the geospatial analysis plot

1. 'import geo_spatial_analysis.py'

2. Check for the dataframe with parking citation data if it has been created
   
3. Need a google client ID for googleMaps to work.

4. call 'geo_spatial_analysis.main(df)', where a is the dataframe consisting of parking citation data

end

# Installations

**Before running this project, please install the following Python libraries:**

- pandas
- numpy
- plotly
- requests
- tqdm
- folium
- seaborn
- matplotlib
- statsmodels
- prophet
- geopy
- prettytable
- googlemaps
- vincent
- branca.colormap
  

**These packages are built-in. No install needed:**

- sys
- ast
- time
- datetime
- collections
- pickle
- calendar
- glob
- os
  

**Versions used:**

- Python 3.11.1
- pandas 1.5.3
- numpy 1.24.1
- plotly 5.13.1
- requests 2.28.2
- tqdm 4.65.0
- folium 0.14.0
- seaborn 0.13.0
- matplotlib 3.8.2
- statsmodels 0.14.0
- prophet 1.1.5
- geopy 2.4.0
- prettytable 3.9.0
- googlemaps 4.10.0
- vincent 0.4
- branca 0.7.0


All of the dependencies can be installed in the terminal using the command:

```
pip install -r requirements.txt
```

# References
**Data:**  
[San Diego Parking Citation Data](https://data.sandiego.gov/datasets/parking-citations/)   
[San Diego Parking Meters Data](https://data.sandiego.gov/datasets/parking-meters-locations/)  
[San Diego Parking Lots Data](https://data.sandiego.gov/datasets/park-locations/)  
[San Diego Population Data](https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html)  
# NEW_143
