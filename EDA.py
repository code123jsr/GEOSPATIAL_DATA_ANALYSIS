# Import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import calendar
from prettytable import PrettyTable
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
        df = pd.read_csv('./Datasets/Parking Ticket Databases/' + f) # make sure to change this to the correct python read path
        df = df.reindex(index=df.index[::-1])
        con_list.append(df)
    for f in read_list_norm:
        df = pd.read_csv('./Datasets/Parking Ticket Databases/' + f) # make sure to change this to the correct python read path
        con_list.append(df)
    
    output_df = pd.concat(con_list)
    return output_df


def top_locations(data, num_top=10):
    """
    This function returns a pie chart for the top ten locations with the most citations.

    Parameters:
    - location_counts (pd.Series): Processed location counts.
    - num_top (int): Number of top locations to display in the pie chart.

    Returns:
             The pie chart
    """
    # Ensure the input DataFrame is not empty
    assert isinstance(data, pd.DataFrame)
    # Convert 'location' column to uppercase
    data['location'] = data['location'].str.upper()

    # Count occurrences of 'vio_desc' and sort in descending order
    location_counts = data['vio_desc'].value_counts().sort_values(ascending=False)
    
    # Ensure the number of top locations is valid
    assert num_top > 0, "Number of top locations should be greater than 0."

    # Select the top locations
    top_locations = location_counts.head(num_top)

    # Plot a pie chart
    plt.pie(top_locations, labels=top_locations.index, autopct='%1.1f%%', startangle=45)
    plt.title('Top Locations with Most Citations')
    return(plt.show())


def locations_by_count(data, num_top=20):
    """
    This function returns the bar plot for the top 20 locations of citation counts.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'location' column.
    - num_top (int): Number of top locations to display in the bar chart.

    Returns:
            The bar plot
    """
    assert isinstance(data, pd.DataFrame)
    # Ensure the number of top locations is valid
    assert num_top > 0, "Number of top locations should be greater than 0."

    # Convert 'location' column to uppercase
    data['location'] = data['location'].str.upper()

    # Calculate location counts and select the top locations
    location_counts = data['location'].value_counts().sort_values(ascending=False).head(num_top)
    location_counts = location_counts.iloc[::-1]

    # Plot a horizontal bar chart
    location_counts.plot(kind='barh', color='red')

    plt.xlabel('Number of Citations', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.yticks(fontsize=10)

    return(plt.show())

def fine_by_description(data, num_top=10):
    """
    This function returns a bar plot of mean violation fine by violation description.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'vio_desc' and 'vio_fine' columns.
    - num_top (int): Number of top violations to display in the bar chart.

    Returns:
             The Bar plot
    """
    assert isinstance(data, pd.DataFrame)
    # Ensure the number of top violations is valid
    assert num_top > 0, "Number of top violations should be greater than 0."

    # Convert 'vio_desc' column to uppercase
    data['vio_desc'] = data['vio_desc'].str.upper()

    # Group by 'vio_desc' and calculate mean fines
    mean_fine_by_desc = data.groupby('vio_desc')['vio_fine'].value_counts().sort_values(ascending=False).head(num_top)
    mean_fine_by_desc = mean_fine_by_desc.iloc[::-1]

    # Plot a horizontal bar chart
    mean_fine_by_desc.plot(kind='barh', color='blue', figsize=(14, 7))

    plt.xlabel('Mean Violation Fine')
    plt.ylabel('Violation Description')
    plt.title('Mean Violation Fine by Violation Description')
    plt.xticks(rotation=90)

    return (plt.show())


def date_by_citation(data, num_top=10):
    """
    This function returns the pie chart of dates with top ten citations count .

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'date_issue' column.
    - num_top (int): Number of top dates to display in the pie chart.

    Returns:
             The pie chart
    """
    assert isinstance(data, pd.DataFrame)
    # Ensure the number of top dates is valid
    assert num_top > 0, "Number of top dates should be greater than 0."

    # Convert 'date_issue' column to datetime
    data['date_issue'] = pd.to_datetime(data['date_issue'])

    # Count occurrences of 'date_issue' and sort in descending order
    daily_counts = data['date_issue'].value_counts().sort_values(ascending=False)

    # Select the top dates
    top_dates = daily_counts.nlargest(num_top)

    # Format day and month labels
    day_month_labels = top_dates.index.strftime('%d-%b')

    # Plot a pie chart
    plt.pie(top_dates, labels=day_month_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Top Dates with Most Citations')
    return (plt.show())

def top_location_monthly_count(data, specific_location):
    """
    This function returns bar plot for number of citations aggrevated for all months for the highest location 

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'location' and 'date_issue' columns.
    - specific_location (str): The specific location for which to analyze monthly citation counts.

    Returns:
              The bar plot
    """
    assert isinstance(data, pd.DataFrame)
    # Ensure the specific location is provided
    assert specific_location, "Specific location is not provided."

    # Convert 'location' column to uppercase
    data['location'] = data['location'].str.upper()

    # Filter data for the specific location
    location_data = data[data['location'] == specific_location].copy()

    # Ensure there are data records for the specific location
    assert not location_data.empty, f"No data found for the specific location: {specific_location}"

    # Convert 'date_issue' column to datetime
    location_data['date_issue'] = pd.to_datetime(location_data['date_issue'])

    # Extract the month from the 'date_issue' and create a new 'month' column
    location_data['month'] = location_data['date_issue'].dt.month_name()

    # Count occurrences of 'month' and reindex to ensure all months are included
    monthly_counts = location_data['month'].value_counts().reindex(calendar.month_name[1:]).fillna(0)

    # Plot a bar chart for monthly citation counts
    monthly_counts.plot(kind='bar', color='red')

    plt.xlabel('Month', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.ylabel('Number of Citations', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.title(f'Monthly Citation Counts for {specific_location}', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.xticks(range(12), calendar.month_abbr[1:], rotation=0)

    return(plt.show())



def top_location_monthly_count(data, specific_location):
    """
    This function returns the bar plot for number of citations aggrevated for all months for the second highest location 

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'location' and 'date_issue' columns.
    - specific_location (str): The specific location for which to analyze monthly citation counts.

    Returns:
            the bar plot
    """
    assert isinstance(data, pd.DataFrame)
    # Ensure the specific location is provided
    assert specific_location, "Specific location is not provided."

    # Convert 'location' column to uppercase
    data['location'] = data['location'].str.upper()

    # Filter data for the specific location
    location_data = data[data['location'] == specific_location].copy()

    # Ensure there are data records for the specific location
    assert not location_data.empty, f"No data found for the specific location: {specific_location}"

    # Convert 'date_issue' column to datetime
    location_data['date_issue'] = pd.to_datetime(location_data['date_issue'])

    # Extract the month from the 'date_issue' and create a new 'month' column
    location_data['month'] = location_data['date_issue'].dt.month_name()

    # Count occurrences of 'month' and reindex to ensure all months are included
    monthly_counts = location_data['month'].value_counts().reindex(calendar.month_name[1:]).fillna(0)

    # Plot a bar chart for monthly citation counts
    monthly_counts.plot(kind='bar', color='red')

    plt.xlabel('Month', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.ylabel('Number of Citations', fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.title(f'Monthly Citation Counts for {specific_location}', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    plt.xticks(range(12), calendar.month_abbr[1:], rotation=0)

    return(plt.show())


def table_counts(data, num_top=10):
    """
    This function displays a table containing violation counts for violation description

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing 'vio_desc' column.
    - num_top (int): Number of top violations to display in the table.

    Returns:
              The table with citation counts for violation description 
    """
    assert isinstance(data, pd.DataFrame)
    # Ensure the number of top violations is valid
    assert num_top > 0, "Number of top violations should be greater than 0."

    # Count occurrences of 'vio_desc' and select the top violations
    vivo_desc_counts = data['vio_desc'].value_counts().sort_values(ascending=False)
    top_vivo_desc = vivo_desc_counts.head(num_top)

    # Create a PrettyTable
    table = PrettyTable()
    table.field_names = ['Violation Description', 'Number of Citations']
    table.border = True
    table.header = True
    table.align = 'l'

    # Add rows to the table
    for vio_desc, count in top_vivo_desc.items():
        table.add_row([vio_desc, count])
    b=table
    return b

from matplotlib.colors import Normalize

def plot_citations_per_year(data, specific_location):
    """
    This function returns the plot for the number of citations per year for a specific location.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing citation data.
    - specific_location (str): Specific location for which to plot citations.

    Returns:
             The bar plot.
    """
    assert isinstance(data, pd.DataFrame)
    # Convert location to uppercase
    specific_location = specific_location.upper()

    # Filter data for the specific location
    location_data = data[data['location'] == specific_location].copy()

    # Convert 'date_issue' to datetime
    location_data['date_issue'] = pd.to_datetime(location_data['date_issue'])

    # Extract year from 'date_issue'
    location_data['year'] = location_data['date_issue'].dt.year

    # Count citations per year and sort by year
    yearly_counts = location_data['year'].value_counts().sort_index()

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define color thresholds and corresponding colors
    color_thresholds = [0, 400, 600, 800, 1000, float('inf')]
    colors = ['blue', 'green', 'yellow', 'orange', 'purple']

    # Assign colors based on citation counts
    color_indices = pd.cut(yearly_counts, bins=color_thresholds, labels=False, right=False)
    color_list = [colors[i] for i in color_indices]

    # Create bar plot
    bars = ax.bar(yearly_counts.index, yearly_counts, color=color_list, edgecolor='red', linewidth=2)

    # Make xlabel and ylabel bold with a border
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')

    # Add a border around xlabel and ylabel
    plt.setp(ax.xaxis.get_label(), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.setp(ax.yaxis.get_label(), bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Set plot title and rotate x-axis labels
    plt.title(f'Number of Cases in {specific_location} by Year', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)

    # Create legend
    legend_labels = ['< 400', '400 - 600', '600 - 800', '800 - 1000', '> 1000']
    legend_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    plt.legend(legend_handles, legend_labels, title='Number of Cases', loc='upper left')

    # Show the plot
    return(plt.show())

if __name__ == "__main__":
    print(top_locations(df))
    print(locations_by_count(df))
    print(fine_by_description(df))
    print(date_by_citation(df))
    print(top_location_monthly_count(df, 'SKI BEACH LOT 3000 INGRAHAM ST'))
    print(top_location_monthly_count(df, '3600 5TH AV'))
    print(table_counts(df))
    print(plot_citations_per_year(df, '3600 5TH AV'))
    

