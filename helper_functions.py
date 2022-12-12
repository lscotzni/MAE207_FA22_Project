from dolfin import *
import numpy as np
from msh2xdmf import import_mesh

def initialize_parameters():
    p = 12
    rpm = 3000
    I_amp = 30
    s = p*3
    Hc = 838.e3

    shift = 15 # degrees
    mech_angle_deg = 0 # degrees
    mech_angle = (shift+mech_angle_deg) * np.pi/180
    elec_angle = 0

    return p, rpm, I_amp, s, Hc, mech_angle, elec_angle

def initialize_geometry(file_path, dim=2, subdomains=True):
    mesh, boundaries_mf, subdomains_mf, ass_table = import_mesh(
        prefix=file_path,
        dim=dim,
        subdomains=subdomains
    )
    dx = Measure('dx', domain=mesh, subdomain_data=subdomains_mf)
    dS = Measure('dS', domain=mesh, subdomain_data=boundaries_mf)

    return mesh, boundaries_mf, subdomains_mf, dx, dS

def get_subdomain_indices():
    iron_sub = [1,2,]
    pm_sub = range(3,14+1)
    winding_sub = range(15,50+1)
    ag_sub = [51,]

    return iron_sub, pm_sub, winding_sub, ag_sub

def initFunctionSpace(mesh, manual_iter=False):
    V_em = FunctionSpace(mesh, 'P', 1) # FUNCTION SPACE FOR EM PDE
    V_t = FunctionSpace(mesh, 'P', 1) # FUNCTION SPACE FOR THERMAL PDE
    V_S = FunctionSpace('R', 0) # FUNCTION SPACE FOR SOURCE TERM (CURRENT DENSITY)
    if manual_iter:
        return V_em, V_t, V_S
    else:
        Ae = FiniteElement('P', mesh.ufl_cell(), 1)
        Te = FiniteElement('P', mesh.ufl_cell(), 1)
        MF = FunctionSpace(mesh, Ae*Te)
        return MF
    
    
