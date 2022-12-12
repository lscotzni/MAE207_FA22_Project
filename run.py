from dolfin import *
import numpy as np
from helper_functions import *
from coupled_pde import CoupledFEAClass

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
Coupled_FEA = CoupledFEAClass(i_amp=I_amp, mesh=mesh, dx=dx, p=p, s=s, Hc=Hc, 
                                mech_angle=mech_angle, elec_angle=elec_angle)

Coupled_FEA.setup_problem()
Coupled_FEA.solve_problem()

vtkfile_A_z = File('solutions/Magnetic_Vector_Potential.pvd')
vtkfile_B = File('solutions/Magnetic_Flux_Density.pvd')
vtkfile_T = File('solutions/Temperature.pvd')
vtkfile_A_z << Coupled_FEA.A_z
vtkfile_B << Coupled_FEA.B
vtkfile_T << Coupled_FEA.T