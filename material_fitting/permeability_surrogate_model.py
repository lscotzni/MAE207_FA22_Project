import numpy as np 
from smt.surrogate_models import RMTB, RBF
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

sns.set()

class PermeabilitySurrogateModel(object):
    def __init__(self):
        pass

    def get_training_data(self):
        self.T_vals = T_vals = np.array([15, 258, 366, 481, 592, 735]) # DEGREES C

        self.H_data = np.array([
            np.array([0.172, 0.344, 0.43, 0.69, 1.075, 1.72, 1.935, 2.15, 2.58, 3.01, 4.30, 5.16, 6.45, 8.60, 12.90, 20.]),
            np.array([0.086, 0.172, 0.344, 0.43, 0.69, 1.075, 1.29, 1.72, 1.935, 2.15, 2.58, 3.44, 4.30, 6.45, 8.60, 20.]), 
            np.array([0.086, 0.172, 0.344, 0.43, 0.69, 1.075, 1.505, 1.935, 2.58, 3.44, 4.30, 5.805, 6.45, 8.60, 20.]), 
            np.array([0.0285, 0.086, 0.172, 0.344, 0.43, 0.69, 1.075, 1.505, 1.935, 2.58, 3.44, 4.30, 5.805, 6.45, 20.]), 
            np.array([0.0285, 0.086, 0.172, 0.344, 0.43, 0.69, 0.86, 1.075, 1.29, 1.505, 1.935, 2.58, 3.44, 4.30, 6.45, 20.]), 
            np.array([0.0285, 0.0568, 0.086, 0.129, 0.172, 0.194, 0.215, 0.258, 0.344, 0.43, 0.69, 1.075, 2.58, 20.])  
        ], dtype=object) * 1.e3 / (np.pi*4) # Oersted, converted to Ampere/meter using this coefficient

        self.mu_data = np.array([
            np.array([461, 543, 630, 814, 1299, 3084, 3245, 3273, 3231, 3129, 2606, 2349, 1969, 1581, 1124, 900]),
            np.array([488, 596, 814, 933, 1315, 2788, 3509, 3931, 3870, 3827, 3609, 3147, 2719, 1980, 1536, 900]),
            np.array([653, 759, 1004, 1150, 2228, 4320, 5078, 4588, 3986, 3304, 2769, 2182, 1988, 1561, 900]),
            np.array([491, 922, 1085, 1628, 1910, 4208, 5743, 5467, 4811, 4070, 3273, 2744, 2109, 1930, 900]),
            np.array([819, 1356, 1790, 3580, 6150, 8204, 7931, 7250, 6504, 5898, 4979, 3944, 3084, 2593, 1863, 900]),
            np.array([2620, 3782, 5101, 9241, 14852, 15660, 16000, 15710, 14442, 12989, 9315, 6300, 2735, 900])
        ], dtype=object) # RELATIVE PERMEABILITY

        vacuum_perm = 4*np.pi*1.e-7

        self.B_data = vacuum_perm*self.mu_data*self.H_data
        num_data_points = 0
        for asdf in range(len(self.mu_data)):
            num_data_points += len(self.mu_data[asdf])

        self.B_training = np.zeros((num_data_points))
        self.mu_training = np.zeros_like(self.B_training)
        self.T_training = np.zeros_like(self.B_training)

        num_pts = 0
        for i, T_val in enumerate(T_vals):
            new_pts = len(self.H_data[i])
            self.B_training[num_pts:num_pts + new_pts] = self.B_data[i]
            self.T_training[num_pts:num_pts + new_pts] = T_val * np.ones_like(self.H_data[i])
            self.mu_training[num_pts:num_pts + new_pts] = self.mu_data[i]

            num_pts += new_pts

        self.training_inputs = np.zeros((self.B_training.shape[0],2))
        self.training_inputs[:,0] = self.B_training
        self.training_inputs[:,1] = self.T_training
        self.input_limits = np.array([[0.0, 5.0], [0.0, 1000]]) # BOUNDS FOR B & T

    def train_model(self):
        print('=== Training Surrogate Model ... ===')
        self.get_training_data()
        # self.sm_mu = RMTB(
        #     xlimits=self.input_limits,
        #     order=3,
        #     num_ctrl_pts=11,
        #     energy_weight=1e-10,
        #     regularization_weight=0.0,
        #     print_global=False,
        #     print_solver=False,
        # )
        self.sm_mu = RBF(d0=1)
        self.sm_mu.set_training_values(self.training_inputs, self.mu_training)
        self.sm_mu.train()

    def get_surrogate_model(self):
        self.train_model()
        return self.sm_mu


if __name__ == '__main__':
    sm_class = PermeabilitySurrogateModel()
    sm_mu = sm_class.get_surrogate_model()

    # print(sm_mu.predict_values(np.array([[2, 100]])))

    plt.figure(1)
    max_mu = np.zeros_like(sm_class.T_vals)
    for i, T_val in enumerate(sm_class.T_vals):
        plt.plot(sm_class.B_data[i], sm_class.mu_data[i], label='T = {}'.format(T_val))
        max_mu[i] = np.max(sm_class.mu_data[i])
        print(np.max(sm_class.mu_data[i]))

    plt.legend()

    # plt.figure(2)
    # for i, T_val in enumerate(sm_class.T_vals):
    #     plt.plot(sm_class.H_data[i], sm_class.mu_data[i], label='T = {}'.format(T_val))
    # plt.legend()

    # plt.figure(3)
    # B_array = np.linspace(0.,3.,150)
    # for i, T_val in enumerate(sm_class.T_vals):
    #     data = []
    #     for j in range(len(B_array)):
    #         data.append([B_array[j], T_val])
    #     data = np.array(data)
    #     # data = np.zeros((2,len(sm_class.B_data[i])))
    #     # data[0,:] = T_val * np.ones_like(sm_class.B_data[i])
    #     # data[1,:] = sm_class.B_data[i]

    #     pred_val = sm_mu.predict_values(data)
    #     plt.plot(B_array,pred_val, label='T = {}'.format(T_val))
    # plt.legend()

    def quadfun(x,a,b,c):
        fun = a*x**2 + b*x + c
        return fun

    p_opt, p_conv = curve_fit(quadfun, sm_class.T_vals[:-1], max_mu[:-1])
    print(p_opt)

    plt.figure(4)
    plt.plot(sm_class.T_vals[:-1], max_mu[:-1], '*-', label='data') # QUADRATIC-LIKE SHAPE; NEGATE LAST VALUE
    plt.plot(sm_class.T_vals[:-1], p_opt[0]*sm_class.T_vals[:-1]**2 + p_opt[1]*sm_class.T_vals[:-1] + p_opt[2], '*-', label='fitting') # QUADRATIC-LIKE SHAPE; NEGATE LAST VALUE
    plt.grid()
    plt.legend()

    plt.show()