# Run this after running stress_sediment.py


# Now update the boundary based on these points
# Let's say that the water surface is now at z=0.55
# We will use y,z as x,y in gmsh, for now.

import gmsh
gmsh.initialize()

gmsh.clear() # While trying this out

y_mesh = np.hstack( (_y[0], _y, _y[-1]) )
z_mesh = np.hstack( ([.6], _eta_1, [.6]) )

# Boundary
for i in range(len(y_mesh)):
    gmsh.model.occ.add_point(y_mesh[i],z_mesh[i],0,tag=i)
point_tags = range(1, i+1)

# Piecewise linear
# Make the line segments
for i in range(len(y_mesh)-1):
    gmsh.model.occ.add_line(i, i+1, tag=i)

# Add one more line segment for the water surface
gmsh.model.occ.add_line(0, i+1, tag=i+1)

#channel_boundary = gmsh.model.occ.add_bspline( point_tags, tag=43 )
#water_surface = gmsh.model.occ.add_bspline( [point_tags[0], point_tags[-1]], tag=44 )
#gmsh.model.occ.add_bezier( list(range(1,5)) ) # Bezier seems to fail with too many points

# Combine these curves into a wireframe boundary
#full_boundary = gmsh.model.occ.add_wire( [channel_boundary, water_surface], tag=45 )
full_boundary = gmsh.model.occ.add_wire( range(i+2), tag=45 )

# Let's see if this way of filling it in works
#mesh_gmsh = gmsh.model.occ.add_bspline_filling( 45, tag=46 )
mesh_gmsh = gmsh.model.occ.add_surface_filling( 45, tag=46 )

# Now synch and display
gmsh.model.occ.synchronize()

# FEniCSx help for creating a mesh
gdim = 2 # I think this is the number of dimensions
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.03)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
gmsh.model.mesh.generate(gdim)

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

