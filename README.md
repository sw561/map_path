# map_path
A python module for plotting maps with various projections.

It also allows for the shortest paths between two points on the globe to be
plotted and for calculation of the path's distance.

Examples of usage are in `paths.py`.

More cities can be added to the file `database`.

The program can easily be extended to include new projections. The projection
class includes methods `xy` and `inverse` which convert between Cartesian
coordinates of the image and longitude and latitude coordinates. These
functions use degrees. Companion functions `_xy` and `_inverse` functions do
the same in terms of radians. By inheriting from one of `Projection_Radians` or
`Projection_Degrees`, only one or other needs to be implemented. To test your
projection, pass an instance of it to `test_projection`.
