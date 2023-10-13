# Python Notes
"""
dictionaries
csvs
collections
default dicts
named touple
datetime
timezone


"""
for index, item in enumerate(list_of_things):
	print(index, item)

#sets
# use set() to create a set
.union()
.intersection()
.difference()


#Dictionaries
.get('key', "Message if it's not found") #is a method that will grab a dictionary entry if it exists
print(names.get(105, "Not Found"))
sorted(names_by_rank, reverse = True)[:10] #prints in decending order
#Keys to a dictionary
print(boy_names.keys())
print(boy_names[2013].keys())
#updating dictionaries
# Assign the names_2011 dictionary as the value to the 2011 key of boy_names
boy_names[2011] = names_2011
# Update the 2012 entry in the boy_names dictionary
boy_names[2012].update([(1, 'Casey'), (2, 'Aiden')])
# Loop over the years in the boy_names dictionary 
for year in boy_names:
    # Sort the data for each year by descending rank and get the lowest one
    lowest_ranked =  sorted(boy_names[year], reverse=True)[0]
    # Safely print the year and the least popular name or 'Not Available'
    print(year, boy_names[year].get(lowest_ranked, 'Not Available')  )



# Remove 2011 from female_names and store it: female_names_2011
female_names_2011 = female_names.pop(2011)

# Safely remove 2015 from female_names with an empty dictionary as the default: female_names_2015
female_names_2015 = female_names.pop(2015, {})

# Delete 2012 from female_names
del female_names[2012]

# Iterate over the 2014 nested dictionary
for rank, names in baby_names[2014].items():
    # Print rank and name
    print(rank, names)


# WORKING WITH CSVs


# Use [1:] to skip the header row
# Import the python CSV module
import csv

# Create a python file object in read mode for the baby_names.csv file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a csv reader on the file object
for row in csv.reader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary
    baby_names[row[5]] = row[3]

# Print the dictionary keys
print(baby_names.keys())

# Import the python CSV module
import csv

# Create a python file object in read mode for the `baby_names.csv` file: csvfile
csvfile = open('baby_names.csv', 'r')

# Loop over a DictReader on the file
for row in csv.DictReader(csvfile):
    # Print each row 
    print(row)
    # Add the rank and name to the dictionary: baby_names
    baby_names[row['RANK']] = row['NAME']

# Print the dictionary keys
print(baby_names.keys())


# COLLECTIONS MODULE
.most_common()
# Import the Counter object
from collections import Counter

# Print the first ten items from the stations list
print(stations[:10])

# Create a Counter of the stations list: station_count
station_count = Counter(stations)

# Find the 5 most common elements
print(station_count.most_common(5))

# Print the station_count
print(station_count)


# Default Dict
from collections import	defaultdict

eateries_by_park = defaultdict(list)
for park_id, name in nyc_eateries_parks:
	eateries_by_park[park_id].append(name)

for eatery in nyc_eateries:
	if eatery.get('phone'):
		eatery_contact_types['phone'] += 1
	if eatery.get('website'):
		eatery_contact_types['websites'] += 1

# Iterate over the entries
for date, stop, riders in entries:
    # Check to see if date is already in the ridership dictionary
    if date not in ridership:
        # Create an empty list for any missing date
        ridership[date] = []
    # Append the stop and riders as a tuple to the date keys list
    ridership[date].append((stop, riders))
    
# Print the ridership for '03/09/2016'
print(ridership['03/09/2016'])

## 
from collections import defaultdict

# Create a defaultdict with a default type of list: ridership
ridership = defaultdict(list)

# Iterate over the entries
for date, stop, riders in entries:
    # Use the stop as the key of ridership and append the riders to its value
    ridership[stop].append(riders)
    
# Print the first 10 items of the ridership dictionary
print(list(ridership.items())[:10])

from collections import OrderedDict
OrderedDict()
nyc_eatery_permits = OrderedDict()
for eatery in nyc_eateries:
	nyc_eatery_permits[eater['end_date']] = eatery
print(list(nyc_eatery_permits.items())[:3])

nyc_eatery_permits.popitem()
nyc_eatery_permits.popitem(last = False) #keyword argument to return items in insertion order

# Named Tuple
from collections import namedtouple
Eatery = namedtouple('Eatery', ['name', 'location', 'park_id','type_name'])
for eatery in nyc_eatery:
	details = Eatery(eatery['name'],
					eatery['location'],
					eatery['park_id'],
					eatery['type_name'])
	eateries.append(details)
print(Eateries[0])

for eatery in eateries[:3]:
	print(eatery.name)
	print(eatery.park_id)
	print(eatery.location)


# Import namedtuple from collections
from collections import namedtuple

# Create the namedtuple: DateDetails
DateDetails = namedtuple('DateDetails', ['date', 'stop', 'riders'])

# Create the empty list: labeled_entries
labeled_entries = []

# Iterate over the entries list
for date, stop, riders in entries:
    # Append a new DateDetails namedtuple instance for each entry to labeled_entries
    labeled_entries.append(DateDetails(date, stop, riders))
    
# Print the first 5 items in labeled_entries
print(labeled_entries[:5])

# Iterate over the first twenty items in labeled_entries
for item in labeled_entries[:20]:
    # Print each item's stop
    print(item.stop)
    # Print each item's date
    print(item.date)
    # Print each item's riders
    print(item.riders)

# DATES AND TIMES
import datetime
.strptime() method convers a string to datetime
date_dt = datetime.strptime(parking_violations_date, '%m/%d/%Y')
.strftime() method converts a string to a datetime
date_dt.strftime('%m/%d/%Y')
isoformat() 

daily_violations = defaultdict(int)
for violation in parking_violations:
	violation_date = datetime.strptime(violation[4], '%m,%d,%Y')
	parking_violations[violation_date.day] += 1
print(sorted(daily_violations.items()))


.now() #returns local datetime on your machine 
.utcnow() #returns the current UTC datetime

now_dt = datetime.now()
utc_dt = datetime.utcnow()

#timezones!
#Naive datetime objects have no timezone data
# Aware datetime objects have a timezone.
# Timezone data is available viq the pytz module via the timezone object
# aware objects have .astimezone() so you can get the time in another time zone.
from pytz import timezone
record_dt = datetime.strptime('07/12/2016 04:39PM', '%m/%d/%Y')
ny_tz = timezone('US/Eastern')
a_tz = timezone('US/Pacific')
ny_dt = record_dt.replace(tzinfor=ny_tz) #makes the naive datetime into aware.
la_dt = ny_dt.astimezone(la_tz)


# adding and subtracting time
timedelta # is used to represent an amount of change in time.
from datetime import timedelta
flashback = timedelta(days = 90)
print(record_dt)
print(record_dt - flashback)
print(record_dt + flashback)

time_diff = record_dt - record2_dt
type(time_diff)
# datetime.timedelta

# Help! Libraries
.parse() will attempt to conver a string to a pendulum datetime object
import pedulum
occured = violation[4] + ' ' + violation[5] + 'M'
occured_dt = predulum.parse(occurred, tz='US/Easter')
print(occured_dt)

.in_timezone() method converts a pendulum time object to a desired timezone.
print(pendulum.now('Asia/Tokyo'))


.in_XXX() # Method procide the differences in a chosen metric
.in_words() # method provides the difference in a nice form

diff = violation_dts[3] - violation_dts[2]
print(diff.in_words())
print(diff.in_days())
print(diff.in_hours())

import pendulum
# Create a now datetime for Tokyo: tokyo_dt
tokyo_dt = pendulum.now('Asia/Tokyo')
# Covert the tokyo_dt to Los Angeles: la_dt
la_dt = tokyo_dt.in_timezone('America/Los_Angeles')=
# Print the ISO 8601 string of la_dt
print(la_dt.to_iso8601_string())


#Chicago Data set
Date, Block, Primary, Type, Description, Location Description, Arrest, Domestic, District
05/23/2016 05:35:00 PM, 024XX W DIVISION ST, ASSULT, SIMPLE, STREET, false, true, 14

from collections import Counter

nyc_eatery_count_by_types = Counter(nyc_eatery_types)

daily_violations = defaultdict(int)

for violation in parking_violations:
	violation_date = datetime.strptime(violation[4], '%m/%d/%Y')
	daily_violations[violation_date.day] += 1


from collections import	defaultdict

eateries_by_park = defaultdict(list)
for park_id, name in nyc_eateries_parks:
	eateries_by_park[park_id].append(name)

print(nyc_eatery_count_by_types.most_common(3))


# PART 2
"""
First determine how many crimes occured by district, and then look at crime by district"""
import csv
csvfile = open('ART_GALLERY.csv', 'r')
for row in csv.DictReader(csvfile):
    print(row)

# pop out a value from the dictionary
galleries_10310 = artgalleries.pop('10310')

# Loop over dictionary
for zip_code, galleries in Art_calleries.items():
    print(zip_code)
    print(galleries)

cookies_eaten_today = ['chocolate chip', 'peanut butter', 'chocolate chip', 'oatmeal cream', 'chocolate chip']
types_of_cookies_eaten = set(cookies_eaten_today)
print(types_of_cookies_eaten)

difference()
cookies_jason_ate.difference(cookies_hugo_ate)
set(['oatmeal cream', 'peanut butter'])










