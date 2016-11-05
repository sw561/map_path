#!/usr/bin/env python

from cities import Cities, Mollweide, Lambert

m = Cities(projection=Mollweide(90))
n = 10

for interm in ["Beijing", "Singapore"]:
    m.plot_path("London",interm,n)
    m.plot_path(interm,"Aukland",n)

m.plot_path("London","Aukland",2*n)

m.savefig("a.png")
