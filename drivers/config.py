cuda = -1

dim = 2
num_nodes_per_element = 3
voigt_map = [[0,1],[2,3]]

fem_dir = '../fem-data'
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# ===================== Material parameters =====================

fem_material = 'ArrudaBoyce' #{'NeoHookeanJ2', 'NeoHookeanJ4', 'Isihara', 'HainesWilson', 'GentThomas','ArrudaBoyce','Ogden'}

# ArrudaBoyce parameters
shear_modulus = 2.5
chain_length = 28
kappa = 1.5

#Ogden parameters
kappa_ogden = 1.5
mu_ogden = 0.65
alpha_ogden = 0.65

noise_level = 'high' # {'none','low','high'}
ensemble_size = 10
output_dir = 'results/'



"""
------------- ICKAN hyperparamters / training settings-------------

ensemble_size:                  Number of ICKANs in the ensemble
random_init: {True,False}       Randomly initialize weights and biases of ICKANs using xavier_uniform
n_input:                        Default = 3, i.e., the three principal invariants.
n_output:                       Default = 1, i.e., the strain energy density.
n_hidden:                       List of number of learnable activations for each hidden layer.
sp_trainable:                   Set the weights for the spline activation to be trainable.
sb_trainable:                   Set the weights for the bias activation to be trainable (does not affect training since bias = zero)
symbolic_enabled:               Enable symbolic regression on the model.
K1_min/max:                     Grid range for the first modified invariant of C.
K2_min/max:                     Grid range for the second modified invariant of C.
K3_min/max:                     Grid range for the third modified invariant of C.
grid:                           Number of grid points.
spline_order:                   Order of the b-splines.
opt_method:                     Specify the NN optimizer.
epochs:                         Number of epochs to train the ICKAN
lr_schedule: {cyclic,multistep} Choose a learning rate scheduler to improve convergence and performance.
eqb_loss_factor:                Factor to scale the force residuals at the free DoFs.
reaction_loss_factor:           Factor to scale the force residuals at the fixed DoFs.
verbose_frequency:              Prints the training progress every nth epoch.
lib:                            Library of monotonic, convex expression for symbolic regression
"""

# =================== ICKAN Parameters ==========================

random_init = True
n_input = 3
n_output = 1
n_hidden = [n_input,2,n_output]
sp_trainable = True
sb_trainable = True
symbolic_enabled = True
K1_min = -5.
K2_min = -5.
K3_min = -5.
K1_max = 25.
K2_max = 25.
K3_max = 25.
grid_range = [[K1_min,K1_max],[K2_min,K2_max],[K3_min,K3_max]]
grid = 12
spline_order = 5
opt_method = 'adam' #{'adam',lbfgs'}
epochs = 1000
lr_schedule = 'cyclic'

if lr_schedule == 'multistep':
    lr = 0.1
    lr_milestones = [250,500,750]
    lr_decay = 0.1
    cycle_momentum = False
elif lr_schedule == 'cyclic':
    base_lr = 0.001
    lr = base_lr
    max_lr = 0.1
    cycle_momentum = False
    step_size_up=50
    step_size_down=50

eqb_loss_factor = 1.
reaction_loss_factor = 1.
verbose_frequency = 1

lib = ['x', 'exp', 'softplus','softplus^2','softplus^3','softplus^4']


"""
-------------Plot settings-------------

plot_quantities: {W,P}          Which quantities to evaluate and plot.
strain_paths:                   Which strain paths to evaluate and plot.
lw_truth:                       Linewidth of the true strain energy density response.
lw_best:                        Linewidth of the strain energy response of the best model.
lw_sym:                         Linewidth of the strain energy response of the symbolic form of the best model.
color_truth:                    Color for the true strain energy reponse
color_best:                     Color for the strain energy response of the best model.
color_sym:                      Color for the strain energy response of the symbolic form of the best model.
alpha_best:                     Opacity of the line for the best model.
alpha_sym:                      Opacity of the line for the symbolic form of the best model.
g_min, g_max:                   Range of the loading parameter gamma.
gamma_steps:                    How many loading steps to take.
fs:                             fontsize
"""


plot_quantities = ['W','P']
material_groups = ['polynomial, special']
strain_paths = ['UT','UC','PS','BC','BT','SS']
lw_truth = 1.
lw_best = 1.
lw_sym = 1.
color_truth = 'black'
color_best = 'red'
color_sym = 'blue'
alpha_sym = 0.25
alpha_best = 0.9
g_min = 0.0
g_max = 2.0
y_min = 0.
gamma_steps = 200
fs = 10


