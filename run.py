from dolfin import *
import numpy as np
from helper_functions import *
from em_pde import *
from thermal_pde import *

import matplotlib.pyplot as plt 
import seaborn as sns
sns.set

''' SET UP BASIC PARAMETERS OF PROBLEM '''
p, rpm, I_amp, s, Hc, mech_angle, elec_angle = initialize_parameters()

''' INITIALIZE GEOMETRY '''
mesh, boundaries_mf, subdomains_mf, dx, dS = initialize_geometry(
    file_path='input_files/motor_mesh_test_1',
    dim=2,
    subdomains=True
)
iron_sd, pm_sd, winding_sd, air_gap_sd = get_subdomain_indices()

''' INITIALIZE PDE DETAILS '''
V_em, V_t, V_S = initFunctionSpace(mesh=mesh)

''' SET UP ELECTROMAGNETIC PDE '''
EMClass = EMClass()
EMClass.setup_EM_pde(V=V_em, i_amp=I_amp, mesh=mesh, dx=dx, p=p, s=s, 
                    Hc=Hc, mech_angle=mech_angle, elec_angle=elec_angle)

''' SET UP THERMAL PDE '''
setup_thermal_pde()

''' RUN NEWTON ITERATION '''
EMClass.solve_EM_pde(mesh)
# B = project(as_vector((solver_ms.A_z.dx(1), -solver_ms.A_z.dx[0])),
#                         VectorFunctionSpace(mesh,'DG',0))

vtkfile_A_z = File('solutions/Magnetic_Vector_Potential.pvd')
vtkfile_B = File('solutions/Magnetic_Flux_Density.pvd')
vtkfile_A_z << EMClass.A_z
vtkfile_B << EMClass.B


# shift = 15 # angular rotor shift in degrees (ccw)
# mech_angles_deg = np.arange(0,30+1,5)
# rotor_rotations = mech_angles_deg[:2]
# instances = len(rotor_rotations)
# mech_angles = (shift+rotor_rotations)*np.pi/180
# # elec_angles = ((shift+rotor_rotations) * np.pi/180) * p/2
# elec_angles = ((rotor_rotations) * np.pi/180) * p/2