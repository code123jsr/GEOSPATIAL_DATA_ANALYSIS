import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pickle
import joblib

def if_holiday(date):
    '''
    :param date: string of a date
    :return: if the date is a holiday
    '''
    assert isinstance(date, pd.Timestamp), 'Invalid Date'

    # if someday is holiday
    us_holidays = {
        'New Years Day': '2023-01-01',
        'Martin Luther King Jr. Day': '2023-01-16',
        'Presidents Day': '2023-02-20',
        'Memorial Day': '2023-05-29',
        'Independence Day': '2023-07-04',
        'Labor Day': '2023-09-04',
        'Thanksgiving Day': '2023-11-23',
        'Christmas Day': '2023-12-25',
        'New Years Day2': '2022-01-01',
        'Martin Luther King Jr. Day2': '2022-01-16',
        'Presidents Day2': '2022-02-20',
        'Memorial Day2': '2022-05-29',
        'Independence Day2': '2022-07-04',
        'Labor Day2': '2022-09-04',
        'Thanksgiving Day2': '2022-11-23',
        'Christmas Day2': '2022-12-25',
    }
    holidays = list(us_holidays.values())

    if str(date)[:10] in holidays:
        return 1
    else:
        return 0



def get_nearby_points_of_interest(lat, lng):
    '''
    :param row: row of df
    :return: if the location is near an interest
    '''
    assert isinstance(lat, float) and isinstance(lng, float), 'Invalid Row'

    # if near an interest
    interests = {
        'balboa park': (32.733742, -117.145545),
        'san diego zoo': (32.735986, -117.150995),
        'old town': (32.754029, -117.196988),
        'seaport village': (32.708990, -117.171087),
        'seaworld sandiego': (32.764251, -117.226386),
        'museum of contemporary art': (32.844553, -117.278324),
        'la jolla cove': (32.850550, -117.273070),
        'blacks beach': (32.880351, -117.251877),
        'mission beach': (32.769585, -117.253014)
    }
    interests_ll = list(interests.values())

    distances = np.sqrt(np.sum((interests_ll - np.array([lat,lng]))**2, axis=1))
    closest_point = interests_ll[np.argmin(distances)]
    if geodesic((lat,lng), closest_point).meters <= 1500:
            return 1
    return 0

def get_address_details_geopy(address):
    '''
    :param address: string of address
    :return: detailed components of address
    '''
    assert isinstance(address, str), "Invalid Input for Address Decomposition"

    address = str(address) + ", San Diego, CA"

    geolocator = Nominatim(user_agent="my_geocoder", timeout=10)
    location = geolocator.geocode(address)
    if location:
        reverse_location = geolocator.reverse((location.latitude, location.longitude), language="en")
        if reverse_location:
            return (location.latitude,location.longitude,reverse_location.raw.get('address', {}).get('postcode'))
    else:
            raise ValueError("Location not found")




def get_nearby_parking(lat, lng):
    '''
    :param row: row of df
    :return: if the location is near a parking meter
    '''
    assert isinstance(lat, float) and isinstance(lng, float), 'Invalid Row'

    meters_location = []

    def get_meters_location(row):
        '''
        :param row: row of DataFrame
        :return: Get all the parking meters' location
        '''
        meters_location.append((row['lat'], row['lng']))

    parking_meters = pd.read_csv("./DataSource/parking_meters_loc_datasd.csv")
    parking_meters.apply(get_meters_location, axis=1)
    meters_location = np.array(list(set(meters_location)))

    distances = np.sqrt(np.sum((meters_location - np.array([lat,lng]))**2, axis=1))
    closest_point = meters_location[np.argmin(distances)]
    if geodesic(closest_point, (lat,lng)).meters < 500:
        return 1
    else:
        return 0


def parking_suggestion(date, location):
    '''
    :param date: query date
    :param location: query location
    :return: suggestion for parking (bool)
    '''
    assert isinstance(date, str) and isinstance(location,str), "Invalid Input for Query"

    with open('suggestion_model.pkl', 'rb') as file:
        mod = pickle.load(file)
    file.close()

    with open('OneHotEncoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    file.close()

    date = pd.to_datetime(date, format = '%Y-%m-%d')
    day, month, weekday = date.day, date.month, date.weekday()

    latitude, longitude, postcode = get_address_details_geopy(location)

    holiday = if_holiday(date)
    interest = get_nearby_points_of_interest(latitude,longitude)
    parking_meter = get_nearby_parking(latitude,longitude)

    days_enc = encoders['days_enc']
    months_enc = encoders['months_enc']
    weekdays_enc = encoders['weekdays_enc']
    postcodes_enc = encoders['postcodes_enc']

    if mod.predict([months_enc.transform([[month]]).toarray().tolist()[0] + days_enc.transform([[day]]).toarray().tolist()[0] + weekdays_enc.transform([[weekday]]).toarray().tolist()[0]+ [holiday] + postcodes_enc.transform([[postcode]]).toarray().tolist()[0] + [latitude] + [longitude] + [interest] + [parking_meter]])[0] == 1:
        print("High Risk")

    else:
        print("Low Risk")


if __name__ == "__main__":
    parking_suggestion("2023-12-7", "200W DATE ST")


