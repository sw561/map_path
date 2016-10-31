#!/usr/bin/env python

from cities import Cities, Equirectangular, Lambert, Mollweide, distance

# Example of using the cities module without plotting (much faster)
# Distance are by default in km unless use_miles is set to True
m = Cities(plot=False, use_miles=1)
print m.distance("London", "Los Angeles")
print m.distance("London", "Beijing")

# Use of miles / km can be switched after initialisation
# m.use_miles = True
# m.use_miles = False

# Cities instances initialised like this can't be used to plot
try:
    m.plot_path("London", "Beijing")
except TypeError as e:
    print e
else:
    print "Didn't catch exception here"

# Exceptions due to cities not being in the database can be caught
from cities import DatabaseKeyError
try:
    print m.distance("London", "Fake")
except DatabaseKeyError as e:
    print e
else:
    print "Didn't catch exception here"

# Example of finding the distance across the surface of a general sphere
# Arguments are points on the surface (longitude, latitude) in degrees
print distance((-30.,10.),(150.,10.),radius=1.0)

for projection in [Equirectangular, Lambert, Mollweide]:

    m = Cities(projection=projection())

    m.plot_path("London", "Los Angeles")
    m.plot_path("London", "New York")
    m.plot_path("Beijing", "New York")
    m.plot_path("Kigali", "Sydney")
    m.plot_path("Quito", "Jakarta")

    # Example of getting coordinates from the database
    # This returns a tuple (longitude, latitude)
    London = m.database.get("London")
    # Example of adding new city to the database
    m.database.add("OppLondon", London[0]+180, -London[1])

    # Example of what will happen if cities are opposite each other
    # The path can not be plotted as the problem is ill conditioned
    m.plot_path("OppLondon", "London")

    m.savefig("%s.png" % projection.__name__)
