import numpy as np
import math
import matplotlib.pyplot as plt
from distutils.version import LooseVersion

# A module to find the shortest path between cities on the globe and plot these
# using various projections.

# Constants
RADIUS_EARTH = 6371. # in km
MILES_PER_KM = 0.621371

def distance(coords1, coords2, radius=RADIUS_EARTH):
    # Just find the distance between two pairs of coordinates across the
    # surface of a sphere. Radius of the sphere defaults to radius of the Earth
    # Format of coords1,2 = (longitude, latitude)
    try:
        disc = Disc(coords1, coords2)
        phi = disc.phi_destination
    except IllConditionedException:
        phi = math.pi
    return phi*radius

def need_plot_setup(f):
    def helper(*args, **kwargs):
        if not args[0].plot:
            s = "Need to use an instance of Cities initialised with plot=True"
            raise TypeError(s)
        return f(*args, **kwargs)
    return helper

class Cities:
    def __init__(self, database_path="database", use_miles=False,
            projection=None, plot=True):
        self.database = Database(database_path)
        self.use_miles = use_miles
        self.plot = plot
        if plot:
            if projection is None:
                projection = Equirectangular()
            self.projection = projection
            self.plotted_cities = set()
            set_up_plot(projection)

    @need_plot_setup
    def plot_city(self, key):
        if key in self.plotted_cities:
            return
        coords = self.database.get(key)
        plot_city(key, coords, self.projection)
        self.plotted_cities.add(key)

    @need_plot_setup
    def plot_path(self, key1, key2, n_points=101):
        for key in (key1, key2):
            self.plot_city(key)

        try:
            disc = self.disc(key1, key2)
        except IllConditionedException:
            print "Not plotting path from %s to %s." % (key1, key2)
            print "Cities on opposite sides of planet.",
            print "Problem is ill-conditioned."
            self.print_distance(key1, key2, math.pi*RADIUS_EARTH)
            return

        plot_path(disc, n_points, self.projection)
        self.print_distance(key1, key2, disc.phi_destination*RADIUS_EARTH)

    def distance(self, key1, key2):
        try:
            disc = self.disc(key1, key2)
            distance = disc.phi_destination*RADIUS_EARTH
        except IllConditionedException:
            distance = math.pi*RADIUS_EARTH
        if self.use_miles:
            distance *= MILES_PER_KM
        return round(distance)

    def disc(self, key1, key2):
        vs = tuple(self.database.get(k) for k in (key1, key2))
        return Disc(vs[0], vs[1])

    def print_distance(self, key1, key2, distance):
        if self.use_miles:
            units = "m"
            distance *= MILES_PER_KM
        else:
            units = "km"
        print "Distance from %s to %s is %.0f%s." % (key1,key2,distance,units)

    @need_plot_setup
    def savefig(self, name, **kwargs):
        savefig(name, **kwargs)

# ------------------------------------------------------------ #
#                                                              #
#               Algebra                                        #
#                                                              #
# ------------------------------------------------------------ #

def cartesian(v):
    # Convert spherical polar vector to Cartesian
    x = v[0]*math.cos(v[1])
    y = x*math.sin(v[2])
    x *= math.cos(v[2])
    z = v[0]*math.sin(v[1])
    v_cart = np.array([x,y,z])
    # assert abs(np.linalg.norm(v_cart)-v[0]) < 1e-8
    return v_cart

def spherical_polar(v):
    # Convert Cartesian vector to spherical polar
    r = math.sqrt(sum(q**2 for q in v))
    phi = math.atan2(v[1], v[0])
    rxy = math.sqrt(v[0]**2+v[1]**2)
    theta = math.atan2(v[2], rxy)
    return np.array([r, theta, phi])

def degrees(angle):
    return angle*180./math.pi

def radians(angle):
    return angle*math.pi/180.

def from_coords(coords):
    # coords is tuple containing (longitude, latitude) in degrees
    (longitude, latitude) = (radians(i) for i in coords)
    cart = cartesian(np.array([1., latitude, longitude]))
    return cart

def coords(vector):
    polar = spherical_polar(vector)
    return tuple(degrees(polar[i]) for i in (2,1))

class IllConditionedException(Exception):
    pass

class Disc(object):
    # To find path start by finding disc which contains centre of the earth,
    # origin and destination. The disc is defined by a normal vector. Get
    # position on edge of disk using azimuthal angle phi.
    #
    # If origin and destination are on same longitude, handle this as a special
    # case, since self.normal will be the zero vector.
    def __init__(self, pos1, pos2):
        # Accept args pos1, pos2 which are tuples (longitude, latitude)
        pos1 = from_coords(pos1)
        pos2 = from_coords(pos2)

        self.normal = np.cross(pos1, pos2)
        # Normalise the normal vector
        normalisation = np.linalg.norm(self.normal)
        if abs(normalisation)<1e-5:
            raise IllConditionedException
        self.normal /= np.linalg.norm(self.normal)

        self.ex = pos1
        self.ey = np.cross(self.normal, self.ex)
        self.phi_destination = math.acos(np.dot(self.ex, pos2))

    def edge(self, phi):
        edge = self.ex*math.cos(phi) + self.ey*math.sin(phi)
        # assert abs(np.dot(edge, self.normal)) < 1e-6
        return coords(edge)

# ------------------------------------------------------------ #
#                                                              #
#               Database                                       #
#                                                              #
# ------------------------------------------------------------ #

class DatabaseKeyError(Exception):
    pass

class Database(object):
    # A custom dict for handling the city coordinate data
    def __init__(self, path):
        f = open(path,"r")
        d = dict()
        for line in f:
            if line[0]=='#': continue
            parts = line.strip().split()
            lon = float(parts[-2])
            lat = float(parts[-1])
            city = " ".join(parts[:-2])
            d[city] = (lon, lat)

        self.d = d

    def get(self, key):
        # Analogous to normal dict get method
        try:
            return self.d[key]
        except KeyError:
            s = "%s not found in database" % key
            raise DatabaseKeyError(s)

    def add(self, name, lon, lat):
        self.d[name] = (lon, lat)

# ------------------------------------------------------------ #
#                                                              #
#          Projections                                         #
#                                                              #
# ------------------------------------------------------------ #

class Projection_Radians(object):
    # For projections implemented using radians
    def xy(self, longitude, latitude):
        return self._xy(radians(longitude), radians(latitude))

    def inverse(self, x, y):
        (lon, lat) = self._inverse(x, y)
        return (degrees(lon), degrees(lat))

class Projection_Degrees(object):
    # For projections implemented using degrees - since both implementations
    # are required by the program
    def _xy(self, longitude, latitude):
        return self.xy(degrees(longitude, degrees(latitude)))

    def _inverse(self, x, y):
        (lon, lat) = self.inverse(x, y)
        return (radians(lon), radians(lat))

class Equirectangular(Projection_Radians):
    def xlimits(self):
        return (-math.pi, math.pi)

    def ylimits(self):
        return (-math.pi/2., math.pi/2.)

    def _xy(self, longitude, latitude):
        return (longitude, latitude)

    def _inverse(self, x, y):
        return (x, y)

class Lambert(Projection_Radians):
    def xlimits(self):
        return (-math.pi, math.pi)

    def ylimits(self):
        return (-1,1)

    def _xy(self, longitude, latitude):
        x = longitude
        y = math.sin(latitude)
        return (x,y)

    def _inverse(self, x, y):
        lon = x
        lat = math.asin(y)
        return (lon,lat)

class Mollweide(Projection_Radians):
    def xlimits(self):
        r2 = math.sqrt(2)
        return (-2*r2, 2*r2)

    def ylimits(self):
        r2 = math.sqrt(2)
        return (-r2, r2)

    def theta(self, phi):
        threshold = 1e-5
        if abs(phi-math.pi/2.)<threshold:
            return phi

        # Theta is the solution to helper(theta) = 0 as follows
        pi_sinphi = math.pi*math.sin(phi)
        def helper(theta):
            return 2.*theta+math.sin(2*theta) - pi_sinphi

        def deriv(theta):
            return 2. + 2.*math.cos(2*theta)

        # Use Newton-Raphson to solve
        theta0 = phi
        counter = 0
        h_theta0 = helper(theta0)
        while abs(h_theta0)>threshold:
            counter += 1
            theta1 = theta0 - h_theta0/deriv(theta0)
            if abs(theta1-theta0)<threshold:
                return (theta0+theta1)/2.
            if counter>100:
                print "DID NOT CONVERGE for phi=%.5e" % phi
                print "Latest guess is %.5e" % theta1
                raise ValueError
            theta0 = theta1
            h_theta0 = helper(theta0)

        return theta0

    def _xy(self, longitude, latitude):
        theta = self.theta(latitude)
        x = 2*math.sqrt(2)/math.pi * longitude * math.cos(theta)
        y = math.sqrt(2) * math.sin(theta)
        return (x,y)

    def _inverse(self, x, y):
        theta = math.asin(y/math.sqrt(2))
        lon = math.pi*x / (2*math.sqrt(2)*math.cos(theta))
        lat = math.asin( (2*theta+math.sin(2*theta))/math.pi )
        return (lon,lat)

def test_projection(projection):
    (xmin, xmax) = projection.xlimits()
    (ymin, ymax) = projection.ylimits()
    for x in np.linspace(xmin, xmax, 11):
        for y in np.linspace(ymin, ymax, 11):
            (lon, lat) = projection.inverse(x, y)
            (xnew, ynew) = projection.xy(lon, lat)
            # print " ".join(["%.5f" for i in range(6)]) %\
            #   (x,y,lon,lat,xnew,ynew)
            assert abs(xnew-x)<1e-5
            assert abs(ynew-y)<1e-5
    print "%s passed projection test." % projection.__class__.__name__

# ------------------------------------------------------------ #
#                                                              #
#          Plotting Routines                                   #
#                                                              #
# ------------------------------------------------------------ #

def get_index(xmin, xmax, nx, x):
    i = (nx-1)*(x-xmin) / (xmax-xmin)
    return int(math.ceil(i))

def set_up_plot(projection):
    fig, ax = plt.subplots()
    fig.set_size_inches(9,5)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.98)
    xmin, xmax = projection.xlimits()
    ymin, ymax = projection.ylimits()
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    ax.set_aspect('equal')

    xt = map(radians,range(-150, 160, 30))
    yt = map(radians,range(-60, 70, 30))
    xvals = map(radians,range(-180, 181, 1))
    yvals = map(radians,range(-90, 91, 1))

    # Plot grid lines
    # Lines of longitude
    for x in xt:
        coords = [projection._xy(x, y) for y in yvals]
        (lon, lat) = zip(*coords)
        plt.plot(lon, lat, color='0.5', linestyle=':')
    # Lines of latitude
    for y in yt:
        coords = [projection._xy(x, y) for x in xvals]
        (lon, lat) = zip(*coords)
        plt.plot(lon, lat, color='0.5', linestyle=':')

    if type(projection) is Equirectangular:
        xt_deg = range(-150, 160, 30)
        yt_deg = range(-60, 70, 30)
        xt = map(radians, xt_deg)
        yt = map(radians, yt_deg)
        plt.xticks(xt, xt_deg)
        plt.yticks(yt, yt_deg)
    else:
        plt.xticks([])
        plt.yticks([])

    for side in ['top','left','right','bottom']:
        ax.spines[side].set_visible(False)

    # Plot land and sea
    if LooseVersion(np.__version__) < LooseVersion('1.7'):
        data = np.load("mapdata.npz")
        mymap = data["data"]
        del data
    else:
        with np.load("mapdata.npz") as data:
            mymap = data["data"]

    if type(projection) is Equirectangular:
        x = np.linspace(xmin, xmax, len(mymap[0]))
        y = np.linspace(ymax, ymin, len(mymap[:,0]))
        z = mymap
    else:
        x = np.linspace(xmin, xmax, 800)
        y = np.linspace(ymin, ymax, 401)
        z = np.zeros([len(y), len(x)])
        # For each z, need to find corresponding point in the map
        xmin = -math.pi
        xmax = math.pi
        ymin = -math.pi/2.
        ymax = math.pi/2.
        for yi in xrange(len(y)):
            for xi in xrange(len(x)):
                # Get lon and lat in radians for efficiency
                (lon, lat) = projection._inverse(x[xi], y[yi])
                if lon<-math.pi or lon>math.pi or lat<-math.pi/2. or lat>math.pi/2.:
                    z[yi,xi] = 0
                else:
                    mlat = -lat
                    map_xi = get_index(xmin, xmax, len(mymap[0]), lon)
                    map_yi = get_index(ymin, ymax, len(mymap[:,0]), mlat)
                    z[yi,xi] = mymap[map_yi,map_xi]

    import matplotlib
    if LooseVersion(matplotlib.__version__) < LooseVersion("1.1.3"):
        plt.pcolor(x,y,z, cmap=plt.cm.Blues)
    else:
        plt.pcolormesh(x,y,z, cmap=plt.cm.Blues, edgecolors='face')
    plt.clim([0,4])

    return fig,ax

def plot_city(name, coords, projection):
    (longitude, latitude) = coords
    (x, y) = projection.xy(longitude, latitude)
    plt.plot(x,y,'go')
    xmin, xmax = projection.xlimits()
    offset = 0.01 * (xmax-xmin)
    plt.annotate(name, xy=(x, y), xytext=(x+offset,y+offset))

def plot_path(disc, n_points, projection):
    # f is a function which returns coordinates for given phi
    phi_points = np.linspace(0, disc.phi_destination, n_points)
    for phi in phi_points:
        point = disc.edge(phi)
        (x, y) = projection.xy(point[0], point[1])
        plt.plot(x,y,'b.')

def savefig(name, **kwargs):
    plt.savefig(name, **kwargs)

if __name__=="__main__":
    test_projection(Equirectangular())
    test_projection(Lambert())
    test_projection(Mollweide())
