#!/usr/bin/env python

from cities import Cities, Equirectangular, Lambert, Mollweide

for projection in [Equirectangular, Lambert, Mollweide]:

    m = Cities(projection=projection())

    m.plot_path("London", "Los Angeles")
    m.plot_path("London", "New York")
    m.plot_path("Beijing", "New York")
    m.plot_path("Kigali", "Sydney")

    # Example of switching units from the default (km)
    # m.use_miles = True
    # m.use_miles = False

    m.plot_path("Quito", "Jakarta")
    m.plot_path("London", "MiamiB")

    # Example of getting coordinates from the database
    # This returns a tuple (longitude, latitude)
    London = m.database.get("London")
    # Example of adding new city to the database
    m.database.add("OppLondon", London[0]+180, -London[1])

    # Example of what will happen if cities are opposite each other
    m.plot_path("OppLondon", "London")

    m.savefig("%s.png" % projection.__name__)
