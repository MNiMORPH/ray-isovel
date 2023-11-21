#! /usr/bin/python3

from mpi4py import MPI
from dolfinx import mesh, fem, plot
from dolfinx.fem import petsc
import numpy as np
import ufl
import pyvista
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
domain = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[ymin, zmin], [ymax, zmax]]), n=[ymax*20+1, zmax*10+1], cell_type=mesh.CellType.quadrilateral)

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
dist_y = ymax - np.abs(x[0])
dist_z = x[1] 
dist = (dist_y**2 + dist_z**2)**.5
rho = 1000. # water density
dudz = 0.01 # Approx as constant for now <-- THIS PART REQUIRES ITERATION !!!!
K = rho * dist * dudz
f = g*S/K
f = dist

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


# Plotting

#pyvista.start_xvfb()
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)

u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()


# Let's put it on a regular grid
# Huh, I'm just redoing what I did below
xyz = np.hstack( [u_geometry[:,:2], np.array([uh.x.array]).transpose()] )
_y = xyz[:,0]
_z = xyz[:,1]
_u = xyz[:,2]
yreg = np.linspace(ymin, ymax, ymax*20*2+1)
zreg = np.linspace(zmin, zmax, zmax*10*2+1)
YREG, ZREG = np.meshgrid(yreg, zreg)

ureg = griddata( np.array([_y, _z]).transpose(), _u,
                 np.array([YREG.ravel(), ZREG.ravel()]).transpose() )
urast = ureg.reshape( (len(zreg), len(yreg)) )


############################
# From "Downslope"

dudz = np.diff(urast, axis=0)
dudy = np.diff(urast, axis=1)

dudz = np.diff( (urast[:,:-1] + urast[:,1:]) / 2., axis=0)
dudy = np.diff( (urast[:-1,:] + urast[1:,:]) / 2., axis=1)

ymid = (yreg[:-1] + yreg[1:])/2.
zmid = (zreg[:-1] + zreg[1:])/2.
umid = (urast[:-1,:-1] + urast[1:,1:])/2.

plt.figure( figsize=(16,2.5))
# Extents from other code
plt.imshow(umid, extent=(ymin, ymax, zmax, zmin))
plt.colorbar( label = 'Flow velocity [m s$^{-1}$]')
plt.ylim(plt.ylim()[::-1])
#sp = plt.streamplot( ymid, zmid, dudy, dudz, density=1, broken_streamlines=False,
#                     color='white' )
#plt.tight_layout()



# Hm, can't use the given lines too well

from streamlines import streamplot2
sl = streamplot2( ymid, zmid, dudy, dudz, density=.7, broken_streamlines=False)
for _sl in sl:
    plt.plot(_sl[:,0], _sl[:,1], linewidth=2, color='1')

# Isovels too

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


plt.xlabel('Lateral distance from channel center [m]')
plt.ylabel('Elevation above bed [m]')

plt.tight_layout()
plt.show()

