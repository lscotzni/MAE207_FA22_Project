from dolfin import *
import numpy as np
from permeability_fitting import *
from fea_utils import *

exp_coeff = extractexpDecayCoeff()
cubic_bounds = extractCubicBounds()

class EMClass(object):
    def __init__(self):
        pass 

    def setup_EM_pde(self, V, i_amp, mesh, dx, p, s, Hc, mech_angle, elec_angle):
        self.A_z = Function(V)
        v = TestFunction(V)
        i_amp = i_amp
        bc =  self.setBCMagnetostatic(V, self.A_z)
        res = self.pdeRes(self.A_z, v, i_amp, dx, p, s, Hc, mech_angle, elec_angle)
        A_z_TF = TrialFunction(V)
        Dres = derivative(res, self.A_z, A_z_TF)
        # Nonlinear solver parameters
        ABS_TOL_M = 1e-6
        REL_TOL_M = 1e-6
        MAX_ITERS_M = 100
        problem_ms = NonlinearVariationalProblem(res, self.A_z,
                                                bc, Dres)
        self.solver_ms = NonlinearVariationalSolver(problem_ms)
        self.solver_ms.parameters['nonlinear_solver']='snes'
        self.solver_ms.parameters['snes_solver']['line_search'] = 'bt'
        self.solver_ms.parameters['snes_solver']['absolute_tolerance'] = ABS_TOL_M
        self.solver_ms.parameters['snes_solver']['relative_tolerance'] = REL_TOL_M
        self.solver_ms.parameters['snes_solver']['maximum_iterations'] = MAX_ITERS_M
        self.solver_ms.parameters['snes_solver']['linear_solver']='mumps'
        self.solver_ms.parameters['snes_solver']['error_on_nonconvergence'] = False
        self.solver_ms.parameters['snes_solver']['report'] = True
        # solver_ms.solve()
        # B = project(as_vector((A_z.dx(1), -A_z.dx[0])),
        #                     VectorFunctionSpace(mesh,'DG',0))

        # return self.solver_ms # want to do the solving in the newton iteration section

    def solve_EM_pde(self, mesh):
        self.solver_ms.solve()
        self.B = project(as_vector((self.A_z.dx(1), -self.A_z.dx(0))),
                        # VectorFunctionSpace(mesh,'DG',0))
                        VectorFunctionSpace(mesh, 'P', 1))

    def setBCMagnetostatic(self, V, A_z):
        A_outer = Constant(0.0)
        outer_bound = OuterBoundary()
        return DirichletBC(V, A_outer, 'on_boundary')

    def RelativePermeability(self, subdomain, u):
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
        ])
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

        '''
        Jm = 0.
        gradv = gradx(v,uhat)
        base_magnet_dir = 2 * np.pi / p / 2
        magnet_sweep    = 2 * np.pi / p
        base_angle      = base_magnet_dir + mech_angle
        # print('base_angle: ', base_angle * 180/np.pi)
        for i in range(p):
            flux_angle = base_angle + i * magnet_sweep
            # print('flux_angle: ', flux_angle * 180/np.pi)
            Hx = (-1)**(i) * Hc * np.cos(flux_angle)
            Hy = (-1)**(i) * Hc * np.sin(flux_angle)

            H = as_vector([Hx, Hy])

            curl_v = as_vector([gradv[1],-gradv[0]])
            Jm += inner(H,curl_v)*dx(i + 2 + 1)
        '''

        num_phases = 3
        num_windings = s
        coil_per_phase = 2
        stator_winding_index_start  = p + 2 + 1
        stator_winding_index_end    = stator_winding_index_start + num_windings
        Jw = 0.
        i_abc = self.compute_i_abc(iq, elec_angle)
        JA, JB, JC = i_abc[0] + DOLFIN_EPS, i_abc[1] + DOLFIN_EPS, i_abc[2] + DOLFIN_EPS

        # NEW METHOD
        # for i in range(int((num_windings) / (num_phases * coil_per_phase))):
        #     coil_start_ind = i * num_phases * coil_per_phase
            
        #     J_list = [
        #         JB * (-1)**(2*i+1) * v * dx(stator_winding_index_start + coil_start_ind),
        #         JA * (-1)**(2*i) * v * dx(stator_winding_index_start + coil_start_ind + 1),
        #         JC * (-1)**(2*i+1) * v * dx(stator_winding_index_start + coil_start_ind + 2),
        #         JB * (-1)**(2*i) * v * dx(stator_winding_index_start + coil_start_ind + 3),
        #         JA * (-1)**(2*i+1) * v * dx(stator_winding_index_start + coil_start_ind + 4),
        #         JC * (-1)**(2*i) * v * dx(stator_winding_index_start + coil_start_ind + 5)
        #     ]
        #     Jw += sum(J_list)

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

    def pdeRes(self, u,v,iq,dx,p,s,Hc,mech_angle,elec_angle):
        """
        The variational form of the PDE residual for the magnetostatic problem
        """
        vacuum_perm = 4e-7 * np.pi
        res = 0.
        num_components = 4 * 3 * p + 2 * s
        for i in range(num_components):
            res += 1./vacuum_perm*(1/self.RelativePermeability(i + 1, u))\
                    *dot(grad(u),grad(v))*dx(i + 1)
        res -= self.JS(v,iq,p,s,Hc,mech_angle, elec_angle)
        return res


# a = (1 / mu)*dot(grad(A_z), grad(v))*dx