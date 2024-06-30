import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import statsmodels.api as sm
import prophet
import seaborn as sn
import datetime as dt

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


def date_reframe(df, date_column, daily = False, monthly = False, yearly = False):
    '''
    :param df: original dataframe of databases
    :param date_column: the column name of date
    :param daily: if need daily count
    :param monthly: if need monthly count
    :param yearly: if need yearly count
    :return: the periodic frequency of the tickets
    '''

    assert isinstance(df, pd.DataFrame) and isinstance(date_column, str) and isinstance(daily, bool) and isinstance(monthly, bool) and isinstance(yearly, bool), "Invalid Input for Date Frequency Converting"

    df.set_index(pd.to_datetime(df[date_column], format='%Y-%m-%d'), inplace=True)
    df = df.sort_index()

    issue_daily, issue_monthly, issue_yearly = defaultdict(list), defaultdict(list), defaultdict(list)
    for index, row in df.iterrows():
        issue_daily[index].append(index)
        issue_monthly[(index.year, index.month)].append(index)
        issue_yearly[index.year].append(index)

    # calculate frequency in different periodic circles
    return_dict = {}

    if daily:
        issue_daily_frequency = defaultdict(int)
        for date, issue in issue_daily.items():
            issue_daily_frequency[date] = len(issue)
        return_dict['daily'] = issue_daily_frequency

    if monthly:
        issue_monthly_frequency = defaultdict(int)
        for date, issue in issue_monthly.items():
            issue_monthly_frequency[date] = len(issue)
        return_dict['monthly'] = issue_monthly_frequency

    if yearly:
        issue_yearly_frequency = defaultdict(int)
        for year, issue in issue_yearly.items():
            issue_yearly_frequency[year] = len(issue)
        return_dict['yearly'] = issue_yearly_frequency

    return return_dict

def regression_visualization(model, x, y, category):
    '''
    :param model: the regression model
    :param x: the original x variable
    :param y: the original y variable
    :param category: the required plot (population/parking meters)
    :return: None (Show Plot)
    '''

    assert len(x) == len(y) and category in ['population', 'parking'], "Invalid Input for Regression Plotting"

    # get predicion
    y_pred = model.predict(x)

    # plot
    plt.scatter([i[1] for i in x], y, label='Scatter Points')
    plt.plot([i[1] for i in x], y_pred, color='orange', label='Linear Fitting')
    if category == 'population':
        plt.xlabel('Population in San Diego')
    else:
        plt.xlabel('Parking in San Diego')
    plt.ylabel('Ticket Amount')
    plt.legend()
    plt.show()


def population_correlation(df, include_2020 = True, visualization = False):
    '''
    :param df: the original DataFrame
    :param include_2020: if include the impact of pandamic (bool)
    :param visualization: if show the visualization (bool)
    :return: P-Value of the regression
    '''

    assert isinstance(df, pd.DataFrame) and isinstance(include_2020, bool) and isinstance(visualization, bool), "Invalid Input for Population Correlation"

    issue_yearly_frequency = date_reframe(df, date_column='date_issue', yearly=True)['yearly']

    # get yearly population of San Diego
    population = pd.read_csv("./Datasets/population_cities_2010_2022.csv", index_col="City")
    sd_cities = ["San Diego city", "Chula Vista city", "Oceanside city", "Escondido city", "Carlsbad city", "El Cajon city", "Vista city", "San Marcos city", "Encinitas city", "National City city"]
    for index, row in population.iterrows():
        if index not in sd_cities: population = population.drop(index)

    # sum the cities of San Diego
    for i in range(2010, 2023):
        population[str(i)] = population[str(i)].str.replace(',', '').astype(float)

    population_whole_sd = [[sum(j for j in population[str(i)])] for i in range(2010, 2023)]


    if include_2020:
        # include the pandemic impact
        x = sm.add_constant([population_whole_sd[i] for i in range(2, len(population_whole_sd))])
        y = [issue_yearly_frequency[i] for i in range(2012, 2023)]
        model = sm.OLS(y, x).fit()
        p_values = model.pvalues


    else:
        # exclude the pandemic impact
        x = sm.add_constant([population_whole_sd[i] for i in range(2, len(population_whole_sd)) if i != 10])
        y = [issue_yearly_frequency[i] for i in range(2012, 2023) if i != 2020]
        model = sm.OLS(y, x).fit()
        p_values = model.pvalues

    if visualization:
        regression_visualization(model, x, y, "population")

    return p_values




def parking_meters_correlation(df, visualization = False):
    '''
    :param df: the original DataFrame
    :param visualization: if show the visualization (bool)
    :return: P-Value of regression
    '''

    assert isinstance(df, pd.DataFrame) and isinstance(visualization, bool), "Invalid Input for Parking Meter Correlation"

    frequency = date_reframe(df, date_column='date_issue', daily=True, monthly=True)

    issue_monthly_frequency = frequency['monthly']

    # get parking meters dataset
    parking_meters = pd.read_csv("./Datasets/parking_meters_loc_datasd.csv")

    new_meters_monthly = date_reframe(parking_meters, date_column='date_inventory', monthly=True)['monthly']

    new_meters_months = sorted(list(new_meters_monthly.keys()), key=lambda item: (item[0], item[1]))
    all_months = sorted(list(issue_monthly_frequency.keys()), key=lambda item: (item[0], item[1]))
    start_position = all_months.index(new_meters_months[0])

    # build variables
    x_meters = sm.add_constant([[new_meters_monthly[all_months[i]]] for i in range(start_position, len(all_months))])
    y_meters = [issue_monthly_frequency[all_months[i]] - issue_monthly_frequency[all_months[i - 1]] for i in range(start_position, len(all_months))]

    model_meters = sm.OLS(y_meters, x_meters).fit()
    p_values_meters = model_meters.pvalues

    if visualization:
        regression_visualization(model_meters, x_meters, y_meters, "parking")

    return p_values_meters


def parking_meters_scatter_plot(df):
    '''
    :param df: the original DataFrame
    :return: the scatter plot of existing parking meterts and changes of parking tickets
    '''

    assert isinstance(df, pd.DataFrame), "Invalid Input for Parking Meters Scatter Plotting"

    issue_daily_frequency = date_reframe(df, date_column='date_issue', daily=True)['daily']

    # get parking meters dataset
    parking_meters = pd.read_csv("./Datasets/parking_meters_loc_datasd.csv")
    new_meters_daily = date_reframe(parking_meters, date_column='date_inventory', daily=True)['daily']

    # align periods between parking meters dataset and parking tickets dataset
    dates = list(new_meters_daily.keys())
    dates.sort()
    sum, current_meters = 0, {}
    for date in dates:
        sum += new_meters_daily[date]
        current_meters[date] = sum

    all_dates = list(issue_daily_frequency.keys())
    all_dates.sort()
    start_point = 0
    for d in all_dates:
        if d in current_meters:
            start_point = all_dates.index(d)
            break

    all_dates = sorted(list(issue_daily_frequency.keys()))[start_point:]

    tickets = [issue_daily_frequency[date] for date in all_dates]

    meters = []
    for date in all_dates:
        if date not in current_meters:
            if len(meters) == 0:
                meters.append(0)
            else:
                meters.append(meters[-1])
        else:
            meters.append(current_meters[date])

    # plot scatter points
    plt.figure(figsize=(20, 10), )
    plt.plot(list(all_dates), np.array(tickets), label='Tickets Amount (Daily)', marker='o')
    plt.plot(list(all_dates), np.array(meters), label='Parking Meters Existed (Daily)', marker='o')
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Amount', fontsize=15)
    plt.legend(fontsize='large')
    plt.xticks(rotation=45)
    plt.show()

# add COVID as additional regressor
def lockdown(ds):
    '''
    :param ds: the date string
    :return: if it is holiday (bool)
    '''

    assert isinstance(ds, pd.Timestamp), "Invalid Input for Lockdown Identify"

    # treat COVID as one-off holidays
    date = pd.to_datetime(ds.strftime('%Y-%m-%d'), format='%Y-%m-%d')

    # lockdown periods
    lockdown_ranges = [
        pd.date_range('2020-03-21', '2020-06-06'),
        pd.date_range('2020-07-09', '2020-10-27'),
        pd.date_range('2021-02-13', '2021-02-17'),
        pd.date_range('2021-05-28', '2021-06-10')
    ]

    lockdown_ranges_set = set(date for lockdown_range in lockdown_ranges for date in lockdown_range)
    if date in lockdown_ranges_set:
        return 1
    else:
        return 0

def seasonal_decompose(df, visualization = False):
    '''
    :param df: original DataFrame
    :param visualization: if show the visualization (bool)
    :return: the seasonal decomposation statistics
    '''

    assert isinstance(df, pd.DataFrame) and isinstance(visualization, bool), "Invalid Input for Seasonal Decompose"

    issue_daily_frequency = date_reframe(df, date_column='date_issue', daily=True)['daily']
    df_issue_daily_frequency = pd.DataFrame(list(issue_daily_frequency.items()), columns=['ds', 'y'])

    # add pandemic as effect
    df_issue_daily_frequency['lockdowns'] = df_issue_daily_frequency['ds'].apply(lockdown)

    # build model
    model = prophet.Prophet()
    model.add_regressor('lockdowns')

    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

    model.add_country_holidays(country_name='US')

    model.fit(df_issue_daily_frequency)
    future = model.make_future_dataframe(periods=365)
    future['lockdowns'] = future['ds'].apply(lockdown)

    forecast = model.predict(future)

    # get decomposation result
    result_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'holidays', 'weekly', 'yearly', 'lockdowns']]
    if visualization:
        model.plot(forecast, xlabel='Date', ylabel='Ticket Amount (Daily)')
        model.plot_components(forecast)

    return result_df



def parse_cols_month(df):
    '''
    Combines all of the month's parking citations.
    df: sorted dataframe from parse_data()
    '''
    assert isinstance(df, pd.DataFrame)

    m = -1
    y = -1
    sum_count = 0
    temp_dict = {}
    for date in df['date_issue']:
        if m == -1 or y == -1:
            m = date.month
            y = date.year
        if date.month == m and date.year == y:
            sum_count += df.loc[date, 'count']
        else:
            temp_dict[str(y) + '-' + str(m)] = sum_count
            m = -1
            y = -1
            sum_count = df.loc[date, 'count']
        if date.month == 10 and date.year == 2023:  # covers edge case on last iteration
            temp_dict[str(2023) + '-' + str(10)] = sum_count
    output_DF = pd.DataFrame()
    output_DF.index = temp_dict.keys()
    output_DF['count'] = temp_dict.values()
    return output_DF


def parse_cols_year(df):
    '''
    Lists the parking citation count by month (rows) and by year (columns).
    df: output dataframe of parse_cols_month(df)
    '''
    assert isinstance(df, pd.DataFrame)

    y = -1
    m_in_same_y = []
    output_DF = pd.DataFrame()
    for date in df.index:
        datee = dt.datetime.strptime(date, "%Y-%m")
        if y == -1:
            y = datee.year
        if datee.year == y:
            m_in_same_y.append(df.loc[date, 'count'])
        else:
            output_DF[str(y)] = m_in_same_y
            y = -1
            m_in_same_y = []
            m_in_same_y.append(df.loc[date, 'count'])
        if datee.month == 10 and datee.year == 2023:  # covers edge case on last iteration, November and December have not passed
            m_in_same_y.append(None)
            m_in_same_y.append(None)
            output_DF[str(2023)] = m_in_same_y
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    output_DF.index = month_names

    return output_DF


def parse_cols_daily(month, df):
    '''
    Cuts out all data besides the ones listed for a particular month.
    month: month to capture data from
    df: dataframe with parking citation data
    '''
    assert isinstance(month, int) and isinstance(df, pd.DataFrame)
    assert 1 <= month <= 12

    temp_dict = {}
    for date in df['date_issue']:
        if date.month == month:
            d = date.day
            y = date.year
            m = date.month
            temp_dict[str(y) + '-' + str(m) + '-' + str(d)] = df.loc[date, 'count']
    output_DF = pd.DataFrame()
    output_DF.index = temp_dict.keys()
    output_DF['count'] = temp_dict.values()
    return output_DF


def parse_cols_daily2(df):
    '''
    Lists the parking citation count by day (rows) and by year (columns).
    df: output dataframe of parse_cols_daily(df)
    '''
    assert isinstance(df, pd.DataFrame)

    y = -1
    d_in_same_y = []
    output_DF = pd.DataFrame()
    month = 0
    for date in df.index:
        month = dt.datetime.strptime(date, "%Y-%m-%d").month
        if y == -1:
            y = dt.datetime.strptime(date, "%Y-%m-%d").year
        if dt.datetime.strptime(date, "%Y-%m-%d").year == y:
            d_in_same_y.append(df.loc[date, 'count'])
        else:
            if len(d_in_same_y) == 28:
                d_in_same_y.append(None)
            output_DF[str(y)] = d_in_same_y
            y = -1
            d_in_same_y = []
            d_in_same_y.append(df.loc[date, 'count'])

    if len(d_in_same_y) == 28:  # to catch edge case of last year
        d_in_same_y.append(None)
    output_DF[str(y)] = d_in_same_y

    if month == 4 or month == 6 or month == 9 or month == 11:
        day_list = range(1, 31)
    elif month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
        day_list = range(1, 32)
    else:
        day_list = range(1, 30)
    output_DF.index = day_list

    return output_DF


def time_heatmap_yearly(df):
    '''
    Parses the dataframe and plots time series heatmaps for the months of October, November, December
    and for all the months over the years.
    df: dataframe with parking citation data
    '''
    assert isinstance(df, pd.DataFrame)

    issue_daily_frequency = pd.DataFrame(list(date_reframe(df, date_column='date_issue', daily=True)['daily'].items()), columns=['date_issue', 'count'])
    issue_daily_frequency.set_index(issue_daily_frequency['date_issue'], inplace=True)


    pcm = parse_cols_month(issue_daily_frequency)
    pcy = parse_cols_year(pcm)
    data_m = pcy.to_numpy()


    x_axis_labels = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    y_axis_labels_12 = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                        'October', 'November', 'December']

    cmap = sn.cm.rocket_r
    # plotting the heatmap for months
    hm = sn.heatmap(data_m, xticklabels=x_axis_labels, yticklabels=y_axis_labels_12, cmap=cmap)
    # displaying the plotted heatmap
    plt.show()



def time_heatmap_monthly(df, month):
    '''
    Parses the dataframe and plots time series heatmaps for the months of October, November, December
    and for all the months over the years.
    df: dataframe with parking citation data
    '''
    assert isinstance(df, pd.DataFrame)

    issue_daily_frequency = pd.DataFrame(list(date_reframe(df, date_column='date_issue', daily=True)['daily'].items()), columns=['date_issue', 'count'])
    issue_daily_frequency.set_index(issue_daily_frequency['date_issue'], inplace=True)

    pcd = parse_cols_daily2(parse_cols_daily(month, issue_daily_frequency))

    data = pcd.to_numpy()

    x_axis_labels = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    y_axis_lables_31 = list(range(1, 32))

    cmap = sn.cm.rocket_r
    # plotting the heatmap for October
    hq = sn.heatmap(data, xticklabels=x_axis_labels, yticklabels=y_axis_lables_31, cmap=cmap)
    # displaying the plotted heatmap
    plt.show()

if __name__ == "__main__":
    df = read_dataset()
    print(population_correlation(df, include_2020=False, visualization=True))
    print(parking_meters_correlation(df, visualization=True))
    parking_meters_scatter_plot(df)
    print(seasonal_decompose(df, visualization=True))
    time_heatmap_yearly(df)
    time_heatmap_monthly(df, 12)


