# Run this after running ray-isovel.py

from scipy.ndimage import gaussian_filter1d

# Plot shear stresses

#plt.figure();
plt.figure( figsize=(16,2.8) )
# How does it compare with Gary's 1.2 prediction?
#plt.plot( y_perim_values,
#          boundary_shear_stress * 1.2 / np.max(boundary_shear_stress),
#          'ko' )
plt.plot( y_perim_values, boundary_shear_stress, 'ko-', linewidth=2 )
#plt.axis('scaled')
plt.xlabel('Lateral distance from channel center [m]', fontsize=18)
plt.ylabel('Boundary\nshear stress [Pa]', fontsize=18)
#plt.colorbar( label = 'Eddy viscosity [m$^2$ s$^{-1}$]') # To take up space
plt.tight_layout()

# Plot sediment transport rates
rho_s = 2650
D = 0.06 # 6 cm gravel

# MPM
tau_star = boundary_shear_stress / ( (rho_s - rho) * g * D )
q_s_star = 3.97 * ( (tau_star > 0.0495) * (tau_star - 0.0495) )**1.5
q_s = q_s_star * ((rho_s-rho)/rho)**.5 * g**.5 * D**1.5

plt.figure( figsize=(16,2.5) )
plt.plot( y_perim_values, q_s, 'o', color='.5',
            label="Sediment discharge per unit width")
plt.xlabel('Lateral distance from channel center [m]', fontsize=18)
plt.ylabel('$q_s$ [m$^2$ s$^{-1}$]', fontsize=18)
#plt.colorbar( label = 'Eddy viscosity [m$^2$ s$^{-1}$]') # To take up space
plt.tight_layout()

# Lateral diffusion of bedload
# It's going to be a related-rates problem, downstream v. cross-stream
# Let's just try a constant value
q_s_diffused = gaussian_filter1d( q_s, 1)
plt.plot( y_perim_values, q_s_diffused, 'o', color='purple', alpha=.5,
          label="Including lateral bed-load diffusion")
plt.legend(fontsize=14)

plt.savefig('qs_diffusion.svg')

# Change in river-bed elevation
_eta_0 = z_perim_values[ z_perim_values == 0 ]
_y = y_perim_values[ z_perim_values == 0 ]
_q_s = q_s[ z_perim_values == 0 ]
_q_s_diffused = q_s_diffused[ z_perim_values == 0 ]

# Constant slope
# Input is diffused, output is local
# AH, THE DX HERE TAKES CARE OF THE RELATED-RATES PROBLEM!
# 4:1 RATIO OF VLEOCITIES
# SO: LET'S SAY 40 METERS UNTIL I FIGURE OUT HOW TO CONVERT U/V RATIO
# TO A SPECIFIC, KNOWN VALUE
# Let's also convert it to days
deta_dt = (1/.65) * (_q_s_diffused - _q_s) / 40 * 86400

# Update after 1 day
_eta_1 = _eta_0 + deta_dt * 1

plt.figure()
plt.plot(_y, _eta_0, color='.5', linewidth=2)
plt.plot(_y, _eta_1, color='purple', alpha=.5, linewidth=2)


# Decreased slope
tau_star = 0.98*boundary_shear_stress / ( (rho_s - rho) * g * D )
q_s_star = 3.97 * ( (tau_star > 0.0495) * (tau_star - 0.0495) )**1.5
q_s = q_s_star * ((rho_s-rho)/rho)**.5 * g**.5 * D**1.5
_q_s = q_s[ z_perim_values == 0 ]
deta_dt = (1/.65) * (_q_s_diffused - _q_s) / 40 * 86400
_eta_1 = _eta_0 + deta_dt * 1

"""
plt.figure()
plt.plot(_y, _eta_0, color='.5', linewidth=4)
plt.plot(_y, _eta_1, color='purple', alpha=.5, linewidth=4)

plt.quiver(_y, _eta_0, (_y-_y), (_eta_1-_eta_0), angles='xy', scale_units='xy',
            scale=1, width=.006, color='.7')
"""

plt.figure( figsize=(16,2.8) )
plt.plot(_y, _eta_0, color='.0', linewidth=4,
          label='Initial channel bed')
plt.plot(_y, _eta_1, color='purple', alpha=.5, linewidth=4,
          label='Evolved channel bed')
plt.quiver(_y, _eta_0, (_y-_y), (_eta_1-_eta_0), angles='xy', scale_units='xy',
            scale=1, width=.003, color='.7')
plt.xlabel('Lateral distance from channel center [m]', fontsize=18)
plt.ylabel('Bed elevation [m]', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
#plt.savefig('aggradation_vectors_20231203.png')

# Increased slope
tau_star = 1.02*boundary_shear_stress / ( (rho_s - rho) * g * D )
q_s_star = 3.97 * ( (tau_star > 0.0495) * (tau_star - 0.0495) )**1.5
q_s = q_s_star * ((rho_s-rho)/rho)**.5 * g**.5 * D**1.5
_q_s = q_s[ z_perim_values == 0 ]
deta_dt = (1/.65) * (_q_s_diffused - _q_s) / 40 * 86400
_eta_1 = _eta_0 + deta_dt * 1


plt.figure( figsize=(16,2.8) )
plt.plot(_y, _eta_0, color='.0', linewidth=4,
          label='Initial channel bed')
plt.plot(_y, _eta_1, color='purple', alpha=.5, linewidth=4,
          label='Evolved channel bed')
plt.quiver(_y, _eta_0, (_y-_y), (_eta_1-_eta_0), angles='xy', scale_units='xy',
            scale=1, width=.003, color='.7')
plt.xlabel('Lateral distance from channel center [m]', fontsize=18)
plt.ylabel('Bed elevation [m]', fontsize=18)
plt.legend(fontsize=14)
plt.tight_layout()
#plt.savefig('incision_vectors_20231203.svg')

"""
plt.figure()
plt.plot(_y, _eta_0, color='.5', linewidth=4)
plt.plot(_y, _eta_1, color='purple', alpha=.5, linewidth=4)

plt.quiver(_y, _eta_0, (_y-_y), (_eta_1-_eta_0), angles='xy', scale_units='xy',
            scale=1, width=.006, color='.7')
"""
