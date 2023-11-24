#! /usr/bin/python3

from mpi4py import MPI
from dolfinx import mesh, fem, plot
from dolfinx.fem import petsc
import numpy as np
import ufl
from skimage import measure

from matplotlib import pyplot as plt
from scipy.interpolate import griddata, interp1d


# DOMAIN AND FUNCTION SPACE
# 8x8 grid
#domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
ymin = -4
ymax = -ymin
zmin = 0
zmax = 1

ny2 = 40
nz = 20

domain = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[ymin, zmin], [ymax, zmax]]), n=[ymax*ny2, zmax*nz], cell_type=mesh.CellType.quadrilateral)

# TRIAL AND TEST FUNCTIONS
V = fem.functionspace(domain, ("Lagrange", 1))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Mixed b.c.
# https://jsdokken.com/dolfinx-tutorial/chapter3/neumann_dirichlet_code.html?highlight=neumann

# Dirichlet
def u_exact(x):
    return 0* x[0] * x[1]

def boundary_D(x):
    return np.isclose(x[1], zmin) + np.isclose(x[0], ymin) + np.isclose(x[0], ymax)

dofs_D = fem.locate_dofs_geometrical(V, boundary_D)
u_bc = fem.Function(V)
u_bc.interpolate(u_exact)
bc = fem.dirichletbc(u_bc, dofs_D)

# Neumann
# I think I need nothing here since it is just 0 gradient
x = ufl.SpatialCoordinate(domain)

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

# SOURCE TERM
# https://fenicsproject.discourse.group/t/how-to-create-a-source-term-that-is-a-spatial-function/11074/6

#p = 1*(1-(x[0]**2+x[1]**2)/1**2)**0.5

# !!!!!!!!!!!!!!!!!!! UPDATE IN HERE FOR SOURCE TERM AS ARRAY !!!!!!!!!!!!!!!!!!
# Could try to write my own algebraic function
# This isn't it, but demonstrates the idea
# Actually, now with rectangle centered in the middle, yes
g = 9.8
S = 1E-2
# Distance from wall as initial guess for the ray-based eddy-size (I think) term
dist_y = ymax - np.abs(x[0])
dist_z = x[1]
dist = (dist_y**2 + dist_z**2)**.5
rho = 1000. # water density
dudz = 0.01 # Approx as constant for now <-- THIS PART REQUIRES ITERATION !!!!
K = rho * dist * dudz
f = g*S/K
f = dist

"""
###############
# ACTIVE WORK #
###############

# SUCCESSFUL TEST. NOW COMMENTING AND RETURNING TO RAY-ISOVEL PROGRESS

# Let's now modify "dist" to match something arbitrary.
# The next step after this is starting with something directly taken
# from an array space and not precalculated

# Coordinates in space
coords = V.tabulate_dof_coordinates()
u2 = fem.Function(V)

# This sets a value in the array
arr = u2.vector.getArray()
arr[0] = 1 # tag
u2.vector.setArray(arr)
print( u2.vector.getArray()[:5] )

# Now, I have to learn what value corresponds to which coordinate
# I think that the values are in the "dof" index order
# But since I have "tabulate dof coordinates", perhaps this order is the same

# IT WORKED! #
# Coordinates are in dof-index order :D 
# Let's test it like this:
u2.vector.setArray(coords[:,0] + coords[:,1])

# And see what it looks like
import pyvista
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = u2.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
u_plotter.show()
############

# Also look into
#u2.vector.setValue() and family


###############
###############
"""

# VARIATIONAL PROBLEM
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
# Dirichlet
#L = f * v * ufl.dx
# Updated with Neumann
# 0 gradient, so g_Neumann = 0
#L = f * v * ufl.dx# - g_Neumann * v * ufl.ds
L = ufl.dot(f,v) * ufl.dx

# SOLVING LINEAR SYSTEM
problem = petsc.LinearProblem(a, L, bcs=[bc],
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Grid directly from FEniCSx
# Values will be exact and should not require interpolation
# Though I keep it here for the more general case of a non-rectangular geometry
coords = V.tabulate_dof_coordinates()
_y = coords[:,0]
_z = coords[:,1]
_u = uh.x.array

# 
ny2_reg_per_meter = 20
nz_reg_per_meter = 20

yreg = np.linspace(ymin, ymax, ymax*ny2_reg_per_meter*2+1)
zreg = np.linspace(zmin, zmax, zmax*nz_reg_per_meter+1)
YREG, ZREG = np.meshgrid(yreg, zreg)

ureg = griddata( np.array([_y, _z]).transpose(), _u,
                 np.array([YREG.ravel(), ZREG.ravel()]).transpose() )
urast = ureg.reshape( (len(zreg), len(yreg)) )


# Extended -- to fit extended diffs
dy = 1/ny2_reg_per_meter
dy2 = dy/2
dz = 1/nz_reg_per_meter
dz2 = dz/2

yreg_ext = np.linspace(ymin-dy2, ymax+dy2, ymax*ny2_reg_per_meter*2+2)
zreg_ext = np.linspace(zmin-dz2, zmax+dz2, zmax*nz_reg_per_meter+2)
YREG_ext, ZREG_ext = np.meshgrid(yreg_ext, zreg_ext)


"""
############################
# From "Downslope"

dudz = np.diff(urast, axis=0)
dudy = np.diff(urast, axis=1)

dudz = np.diff( (urast[:,:-1] + urast[:,1:]) / 2., axis=0)
dudy = np.diff( (urast[:-1,:] + urast[1:,:]) / 2., axis=1)

ymid = (yreg[:-1] + yreg[1:])/2.
zmid = (zreg[:-1] + zreg[1:])/2.
umid = (urast[:-1,:-1] + urast[1:,1:])/2.
"""

# Extend the velocity raster so the rays go farther and cross the isovels
# Just assume same values extend 1 cell beyond bounds -- straight on boundaries
# Should be 0s at walls, same as others at surface
# And it seems to be at 0 or very very very close, so just to make it exactly 0
# I force the issue (seems to reach 0 at banks but not bed in the test case)
# "1*" to indicate top boundary (open channel)
urast_ext = np.vstack( [0*urast[0,:], urast, 1*urast[-1,:]] )
urast_ext = np.column_stack( [0*urast_ext[:,0], urast_ext, 0*urast_ext[:,-1]] )

dudz_ext = np.diff( (urast_ext[:,:-1] + urast_ext[:,1:]) / 2., axis=0)
dudy_ext = np.diff( (urast_ext[:-1,:] + urast_ext[1:,:]) / 2., axis=1)

# Not extended, for plotting
#ymid = (yreg_ext[:-1] + yreg_ext[1:])/2.
#zmid = (zreg[:-1] + zreg_ext[1:])/2.
#umid = (urast_ext[:-1,:-1] + urast_ext[1:,1:])/2.


plt.figure( figsize=(16,2.5) )
# Extents from other code
plt.imshow(urast, extent=(ymin-dy2, ymax+dy2, zmax+dz2, zmin-dz2))
plt.colorbar( label = 'Flow velocity [m s$^{-1}$]')
plt.ylim(plt.ylim()[::-1])
plt.xlim((ymin,ymax))
plt.ylim((zmin,zmax)) # Comment to check: there is a ray on each side at z=1
#sp = plt.streamplot( ymid, zmid, dudy, dudz, density=1, broken_streamlines=False,
#                     color='white' )
#plt.tight_layout()



# Hm, can't use the given lines too well

from streamlines import streamplot2

# Set streamline start points along perimeter
# I bet I can use interp1D for this
# After going through all of the vertices of the channel margin

# FIRST TEST: HARD CODE RECTANGLE
s_perim = [0, zmax, zmax + 2*ymax, zmax + 2*ymax + zmax]
y_perim = [ymin, ymin, ymax, ymax]
z_perim = [zmax, 0, 0, zmax]

f_y_interp = interp1d(s_perim, y_perim)
f_z_interp = interp1d(s_perim, z_perim)

s_perim_values = np.linspace(0, np.max(s_perim), 41)
y_perim_values = f_y_interp(s_perim_values)
z_perim_values = f_z_interp(s_perim_values)

start_points = np.vstack(( y_perim_values, z_perim_values )).transpose()

sl = streamplot2( yreg_ext, zreg_ext, dudy_ext, dudz_ext, broken_streamlines=False, start_points=start_points)

# Arrays are (y,z) and at wall
# So I could trace along outer wall.
# But they should all reach the top free surface in order -- easier!
sl = sorted(sl, key=lambda _sl: _sl[-1,0])

# Hm, doesn't quite seem to work. Maybe they are just too close and precision becomes an issue.
# Go around boundary
# Yes, this works.
sl = sorted( sl, key=lambda _sl: _sl[0,0] + _sl[0,1]*np.sign(_sl[0,0]) )


# Plot
for _sl in sl:
    plt.plot(_sl[:,0], _sl[:,1], linewidth=2, color='1.')


"""
# Rescale
# use ymin, ymax, xmin, xmax, defined in other script
#fig, ax = plt.subplots()
_ep = 1E-3 # small but not miniscule value
contours = []
for _level in np.linspace( np.min(urast) + _ep, np.max(urast) - _ep, 10 ):
    _contours_local = (measure.find_contours(urast, level=_level))
    for __contour in _contours_local:
        __contour[:,0] = __contour[:,0]/urast.shape[0] * (zmax-zmin) + zmin
        __contour[:,1] = __contour[:,1]/urast.shape[1] * (ymax-ymin) + ymin
    # Plot all contours found
    contours.append(_contours_local)
    for contour in contours[-1]:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='.7')
"""

# Use extended urast so contours don't end before velocity raster's end
_ep = 2E-2 # small but not miniscule value
# NOTE: SHOULD SET UP 0s ALONG NO SLIP MARGINS, FOR INTERPOLATION
contours = []
# Rescaling
dy_rast = 2*ymax/urast.shape[1]
ymin_ext = ymin - dy_rast
ymax_ext = ymax + dy_rast
dz_rast = zmax/urast.shape[0]
zmin_ext = zmin - dz_rast
zmax_ext = zmax + dz_rast

zmax_ext_2 = zmax + 2*dz_rast

# Extend raster further to extend isovel lines on top
urast_ext_top2 = np.vstack(( urast_ext[0,:], urast_ext ))

for _level in np.linspace( np.min(urast_ext) \
                            + _ep, np.max(urast_ext) - _ep, 10 ):
    _contours_local = measure.find_contours( urast_ext_top2, level=_level,
                                             fully_connected='high')
    for __contour in _contours_local:
        __contour[:,0] = __contour[:,0]/urast_ext_top2.shape[0] * \
                                              (zmax_ext_2-zmin_ext) + zmin_ext
        __contour[:,1] = __contour[:,1]/urast_ext_top2.shape[1] * \
                                              (ymax_ext-ymin_ext) + ymin_ext
    # Plot all contours found
    contours.append(_contours_local)
    for contour in contours[-1]:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='.7')


plt.xlabel('Lateral distance from channel center [m]')
plt.ylabel('Elevation above bed [m]')


plt.tight_layout()


##########################################
# AWAY FROM PLOTTING: LINE INTERSECTIONS #
##########################################


# Intersection points
# Try with shapely
from shapely.geometry import LineString, Polygon

"""
_ray = LineString( sl[5] )
_isovel = LineString( np.vstack((contours[3][0][:,1], contours[3][0][:,0] )).transpose() ) # Find a way around this "0"
#line.intersection( LineString( 

#line = LineString([(0, 0), (2, 2)])
#inter = line.intersection( LineString([(1, 1), (3, 3)]) )

intersect = _ray.intersection(_isovel)

# Single line pair
plt.figure()
plt.plot(sl[5][:,0], sl[5][:,1])
plt.plot(contours[3][0][:,1], contours[3][0][:,0])
plt.plot(intersect.coords.xy[0], intersect.coords.xy[1], 'ko')
"""


# Multiple pairs
from shapely.geometry import MultiLineString

_rays = MultiLineString(sl)

# The first (full-area) loop need be run only once.
# It follows the intersections of the isovels with the bed
_boundary_coords = np.array([y_perim, z_perim]).transpose()
boundary = LineString(_boundary_coords)

intersections = _rays.intersection( boundary )

_xyinter = []
for _intersect in intersections.geoms:
    _xyinter.append( [_intersect.coords.xy[0][0], _intersect.coords.xy[1][0]] )
_xyinter = sorted(_xyinter, key=lambda _i: _i[0] + _i[1]*np.sign(_i[0]))

if len(_xyinter) != len(sl):
    print( "Intersection error." )
    sys.exit(2)

# Loop from the first to the second to last
# (Need space on the sides to compute areas)
#for i in range(1, len(sl)-1):
    
        



# Then with one isovel
intersect = _rays.intersection(_isovel)
# Iterate through it
# And realize that they come unsorted. Blah.
_xyinter = []
for _intersect in intersect.geoms:
    # Intersect y, z
    _xyinter.append( [_intersect.coords.xy[0][0], _intersect.coords.xy[1][0]] )
_xyinter = sorted(_xyinter, key=lambda _i: _i[0] + _i[1]*np.sign(_i[0]))

if len(_xyinter) != len(sl):
    print( "Intersection error." )
    sys.exit(2)

# NOW: We have all of the intersections

# At this point, let's walk along the isovels to solve for K




plt.show()

