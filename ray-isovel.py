#! /usr/bin/python3

from mpi4py import MPI
from dolfinx import mesh, fem, plot
from dolfinx.fem import petsc
import numpy as np
import ufl
from skimage import measure

from matplotlib import pyplot as plt
from scipy.interpolate import griddata


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
plt.ylim((zmin,zmax))
#sp = plt.streamplot( ymid, zmid, dudy, dudz, density=1, broken_streamlines=False,
#                     color='white' )
#plt.tight_layout()



# Hm, can't use the given lines too well

from streamlines import streamplot2
sl = streamplot2( yreg_ext, zreg_ext, dudy_ext, dudz_ext, density=.7, broken_streamlines=False)
for _sl in sl:
    plt.plot(_sl[:,0], _sl[:,1], linewidth=2, color='1')

# Isovels too

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
for _level in np.linspace( np.min(urast_ext) \
                            + _ep, np.max(urast_ext) - _ep, 10 ):
    _contours_local = (measure.find_contours(urast_ext, level=_level))
    for __contour in _contours_local:
        __contour[:,0] = __contour[:,0]/urast_ext.shape[0] * \
                                              (zmax_ext-zmin_ext) + zmin_ext
        __contour[:,1] = __contour[:,1]/urast_ext.shape[1] * \
                                              (ymax_ext-ymin_ext) + ymin_ext
    # Plot all contours found
    contours.append(_contours_local)
    for contour in contours[-1]:
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color='.7')


plt.xlabel('Lateral distance from channel center [m]')
plt.ylabel('Elevation above bed [m]')

plt.tight_layout()
plt.show()

