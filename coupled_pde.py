from dolfin import *
import numpy as np
from material_fitting.permeability_fitting import *

exp_coeff = extractexpDecayCoeff()
cubic_bounds = extractCubicBounds()

class OuterBoundary(SubDomain):
    """
    Define the subdomain for the outer boundary
    """
    def inside(self, x, on_boundary):
        tol = 1E-8
        r = np.sqrt(x[0]**2 + x[1]**2)
        # return on_boundary and near(r, 0.12, tol)# 0.12: outer radius
        return on_boundary and r > 0.09 # 0.12: outer radius
        # return on_boundary # 0.12: outer radius
        # return on_boundary and near(r,0,tol) # 0.12: outer radius

class CoupledFEAClass(object):
    def __init__(self, i_amp, mesh, dx, p, s, Hc, mech_angle, elec_angle):
        self.i_amp = i_amp
        self.mesh = mesh
        self.dx = dx
        self.p = p
        self.s = s 
        self.Hc = Hc
        self.mech_angle = mech_angle 
        self.elec_angle = elec_angle
        self.l_ef = 70.e-3 # effective length
        self.rpm = 1000

        self.winding_area = 20 * (103+83)/2*np.pi/36 * 1e-6 # m^2
        self.num_components = 4 * 3 * p + 2 * s
        self.f = self.rpm/(2*60) * self.p

    def setup_function_spaces(self):
        Ae = FiniteElement('P', self.mesh.ufl_cell(), 1)
        Te = FiniteElement('P', self.mesh.ufl_cell(), 1)
        self.MFS = FunctionSpace(self.mesh, MixedElement([Ae,Te]))
    
    def setBC_EM(self):
        bc = DirichletBC(self.MFS.sub(0), Constant(0.0), 'on_boundary')
        return bc

    def assemble_EM_res(self, A_z, T, v, i_amp, dx, p, s, Hc, mech_angle, elec_angle):
        vacuum_perm = 4e-7 * np.pi
        res = 0.
        for i in range(self.num_components):
            res += 1./vacuum_perm*(1/self.RelativePermeability(i + 1, A_z, T))\
                    *dot(grad(A_z),grad(v))*dx(i + 1)
        res -= self.JS(v,i_amp,p,s,Hc,mech_angle, elec_angle)
        return res

    def assemble_Temp_res(self, A_z, u, T, v, dx, p, s, alpha=25.32, T_atm=70):
        res = 0
        for i in range(self.num_components):
            res += self.ThermalConductivity(i+1, T)*dot(grad(T),grad(v))*dx(i + 1)

        # apply Robin BC here
        boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)
        outer_bound = OuterBoundary()
        outer_bound.mark(boundary_markers, 1)
        ds = Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)
        # USE ds(1) to mark the outer boundary for the PDE
        res += alpha*v*(T - T_atm)*ds(1)

        res -= self.QS(self.i_amp, A_z, u, T, v, self.dx, self.elec_angle, self.p)
        return res
        

    def setup_problem(self):
        self.setup_function_spaces()
        self.MF = Function(self.MFS)
        self.A_z, self.T = split(self.MF) # splitting function space into 2 functions
        (u, v) = TestFunctions(self.MFS)

        ''' EM PDE '''
        EM_bc = self.setBC_EM()
        EM_res = self.assemble_EM_res(self.A_z, self.T, u, self.i_amp, self.dx, self.p, self.s,
                                    self.Hc, self.mech_angle, self.elec_angle)
        # print(type(EM_res))
        # exit()
        # V = FunctionSpace(self.mesh, 'P', 1) # FUNCTION SPACE FOR THERMAL PDE
        # self.T = Function(V)
        # v = TestFunction(V)

        ''' Temperature PDE '''
        Temp_res = self.assemble_Temp_res(self.A_z, u, self.T, v, self.dx, self.p, self.s)
        # boundary_markers = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)
        # outer_bound = OuterBoundary()
        # outer_bound.mark(boundary_markers, 1)
        # bc = DirichletBC(V, Constant(70), outer_bound)
        # a, L = self.assemble_Temp_res(self.T, v, self.dx, self.p, self.s)
        # a, L  = lhs(Temp_res), rhs(Temp_res)
        # self.T = Function(V)
        # solve(a == L,self.T)
        # solve(Temp_res  == 0, self.T)
        # vtkfile_T = File('solutions/Temperature.pvd')
        # vtkfile_T << self.T
        # exit()

        self.res = EM_res + Temp_res
        # self.res = Temp_res

        # SETTING UP SOLVER
        '''
        NEED:
            - TRIAL FUNCTIONS
            - DERIVATIVES
        '''
        # A_z_T, T_T = TrialFunctions(self.MF)
        # solve(self.res == 0, self.MF)
        if True:
            Dres = derivative(self.res, self.MF, TrialFunction(self.MFS))
            ABS_TOL_M = 1e-6
            REL_TOL_M = 1e-6
            MAX_ITERS_M = 100
            problem = NonlinearVariationalProblem(self.res, self.MF, EM_bc, Dres)
            self.solver = NonlinearVariationalSolver(problem)
            self.solver.parameters['nonlinear_solver']='snes'
            self.solver.parameters['snes_solver']['line_search'] = 'bt'
            self.solver.parameters['snes_solver']['absolute_tolerance'] = ABS_TOL_M
            self.solver.parameters['snes_solver']['relative_tolerance'] = REL_TOL_M
            self.solver.parameters['snes_solver']['maximum_iterations'] = MAX_ITERS_M
            self.solver.parameters['snes_solver']['linear_solver']='mumps'
            self.solver.parameters['snes_solver']['error_on_nonconvergence'] = False
            self.solver.parameters['snes_solver']['report'] = True
        
    def solve_problem(self):
        self.solver.solve()
        self.A_z, self.T = self.MF.split()
        self.B = project(as_vector((self.A_z.dx(1), -self.A_z.dx(0))),
                    VectorFunctionSpace(self.mesh, 'DG', 0))

    ''' EM FUNCTIONS '''
    def RelativePermeability(self, subdomain, u, T):
        # if subdomain == 1: # Electrical/Silicon/Laminated Steel
        if subdomain == 1 or subdomain == 2: # Electrical/Silicon/Laminated Steel
            B = as_vector([u.dx(1), -u.dx(0)])
            norm_B = sqrt(dot(B, B) + DOLFIN_EPS)

            mu = conditional(
                lt(norm_B, cubic_bounds[0]),
                linearPortion(norm_B),
                conditional(
                    lt(norm_B, cubic_bounds[1]),
                    cubicPortion(norm_B),
                    (exp_coeff[0] * exp(exp_coeff[1]*norm_B + exp_coeff[2]) + 1)
                )
            )
            # mu = 5500
            # [ 1.71232761e-02 -2.25810238e+00  3.33894181e+03]
            # FITTING OF MU AS A FUNCTION OF TEM
            mu = mu * (1.71232761e-02*T**2 -2.25810238e+00*T + 3.33894181e+03)/4000
            # 4000 is max relative permeability for baseline in conditional statement above
        elif subdomain >= 3 and subdomain <= 14: # NEODYMIUM
            mu = 1.04457 # insert value for titanium or shaft material
        elif subdomain >= 15 and subdomain <= 50: # COPPER
            mu = 0.999
        elif subdomain >= 51: # AIR
            mu = 1.00

        return mu

    def compute_i_abc(self, iq, angle=0.0):
        i_abc = as_vector([
            iq * np.sin(angle),
            iq * np.sin(angle - 2*np.pi/3),
            iq * np.sin(angle + 2*np.pi/3),
        ]) / self.winding_area
        return i_abc

    def JS(self, v,iq,p,s,Hc,mech_angle,elec_angle):
        """
        The variational form for the source term (current) of the
        Maxwell equation
        """
        Jm = 0.
        base_magnet_dir = 2 * np.pi / p / 2
        magnet_sweep    = 2 * np.pi / p
        base_angle = base_magnet_dir + mech_angle
        for i in range(p):
            flux_angle = base_angle + i * magnet_sweep
            Hx = (-1)**(i) * Hc * np.cos(flux_angle)
            Hy = (-1)**(i) * Hc * np.sin(flux_angle)

            H = as_vector([Hx, Hy])

            curl_v = as_vector([v.dx(1),-v.dx(0)])
            Jm += inner(H,curl_v)*dx(i + 2 + 1)

        num_phases = 3
        num_windings = s
        coil_per_phase = 2
        stator_winding_index_start  = p + 2 + 1
        stator_winding_index_end    = stator_winding_index_start + num_windings
        Jw = 0.
        i_abc = self.compute_i_abc(iq, elec_angle)
        JA, JB, JC = i_abc[0] + DOLFIN_EPS, i_abc[1] + DOLFIN_EPS, i_abc[2] + DOLFIN_EPS

        coils_per_pole  = 3
        for i in range(p): # assigning current densities for each set of poles
            coil_start_ind  = stator_winding_index_start + i * coils_per_pole
            coil_end_ind    = coil_start_ind + coils_per_pole

            J_list = [
                JB * (-1)**(i+1) * v * dx(coil_start_ind),
                JA * (-1)**(i) * v * dx(coil_start_ind + 1),
                JC * (-1)**(i+1) * v * dx(coil_start_ind + 2),
            ]

            Jw += sum(J_list)

        return Jm + Jw

    def ThermalConductivity(self, subdomain, T):
        if subdomain == 1 or subdomain == 2: # Electrical/Silicon/Laminated Steel
            k = 30
            # k = 54 - T * 3.33e-2
        elif subdomain >= 3 and subdomain <= 14: # NEODYMIUM
            k = 9 
        elif subdomain >= 15 and subdomain <= 50: # COPPER
            # LINEAR FITTING OF FORM k = a*T + b;
            # [a,b] = [-6.31118918e-02,  4.01904970e+02] from copper_fitting.py
            k = -6.31118918e-02 * T + 4.01904970e+02
            # k = 400
        elif subdomain == 51: # AIR GAP
            eta = 81./80. # stator inner / rotor outer (mm/mm)
            nu = 15.67e-6 # kinematic viscosity of air (m^2/s)
            delta = 1e-3 # air gap thickness (m) 
            omega = 1000 * 2. * np.pi / 60. * 80e-3 # tangential velocity @ rotor (m/s)
            Re = omega*delta/nu
            k = 0.0019 * eta**(-2.9084) * Re**(0.4614*ln(3.33361*eta))
            # NOTE: k ~ 0.0622 with equation above in air gap @ 1000 RPM
            # k = 10
        elif subdomain > 51: # AIR SLOTS
            k = .02662

        return k

    def AvgCopperResistance(self, subdomain, T_avg=20):
        l_ef = 70.e-3 # effective length
        # rho = 0.02716e-6 # resistivity
        rho = 1.724e-8
        N = 12 # num windings
        A = self.winding_area # winding area
        R = N**2*l_ef/A * rho * (1. + (T_avg - 20.)) # 
        # R = N**2*l_ef/A * rho * (T_avg + (- 20. + 1))
        # print(R)
        # print(R*dx(subdomain))
        # print(type(R*dx(subdomain)))
        # print(assemble(R*dx(subdomain)))
        # exit()
        return R
    
    def AvgWindingTemperature(self, function, subdomain):
        # Need to integrate/average temperature over the subdomain
        # print(type(self.MF))
        # print(type(self.MF.sub(1)))
        # print(self.MF.sub(1))
        # exit()

        _, T = self.MF.split()
        func_unit = interpolate(Constant(1.0), self.MF.sub(1).function_space().collapse())

        integral = inner(function, func_unit)
        # print(integral)
        # print(assemble(integral*dx(subdomain)))
        
        # area = self.getSubdomainArea(subdomain)
        # avg_func = assemble(integral)

        # avg_func = assemble(integral)/assemble(self.winding_area)
        return integral

    def QS(self, i_amp, A_z, u, T, v, dx, elec_angle, p):
        q = 0
        # CORE LOSSES
        C = 20 # PLAY AROUND WITH THIS VALUE TO SHOW DIFFERENCE IN OUTPUT
        B_magnitude = sqrt(A_z.dx(0)**2 + A_z.dx(1)**2)
        for i in [1,2]:
            print(type(pow(B_magnitude,2)))
            print(type(pow(B_magnitude,2)*self.dx(i+1)))
            
            # ec_loss = 2.*np.pi**2*self.f**2*self.l_ef*0.07*v*pow(B_magnitude,2)*self.dx(i)
            ec_loss = C * 2.*np.pi**2*self.f**2*self.l_ef*0.07*v*self.dx(i)
            q += ec_loss

            # h_loss = 55.*2.*np.pi*self.f*self.l_ef*v*self.dx(i)*pow(B_magnitude,1.76835)
            h_loss = C * 55.*2.*np.pi*self.f*self.l_ef*v*self.dx(i)
            q += h_loss
            

        # COPPER LOSSES
        winding_subdomain_start = 15
        i_abc = self.compute_i_abc(i_amp, elec_angle)
        for i in range(p):
            T_avg = self.AvgWindingTemperature(function=T, subdomain=15+3*i)
            # print(T_avg)
            print(type(T_avg))
            T_avg = 20
            R = self.AvgCopperResistance(subdomain=15+3*i, T_avg=T_avg)
            # R = self.AvgCopperResistance()
            # print(R)

            # q += Constant(R) * v*(Constant(i_abc[1]**2)*dx(15+3*i) + Constant(i_abc[0]**2)*dx(15+3*(i+1)) + \
            #     Constant(i_abc[2]**2)*dx(15+3*(i+2)))
            q += 12 * R * v * self.winding_area * (i_abc[1])**2 * dx(15+3*i) + \
                12 * R * v * self.winding_area * (i_abc[0])**2 * dx(15+3*i+1) + \
                12 * R * v * self.winding_area * (i_abc[2])**2 * dx(15+3*i+2)
            # q += 12. * self.winding_area * R * (i_abc[1])**2 * v * dx(15+3*i)
        
        return q
