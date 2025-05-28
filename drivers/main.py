#=====================================================================
# INITIALIZATIONS:
#=====================================================================
#sys and core
import sys
sys.path.insert(0, '../')
from core import *
#config
from config import *
#CUDA
initCUDA(cuda)
#supporting files
from model import *
from train import *
from helper import *
from post_process import *

datasets = []

fem_material = sys.argv[1]
noise_level = sys.argv[2]

loadsteps = []
if 'NeoHookean' in fem_material:
    loadsteps = [10,20,30]
elif fem_material == 'Ogden':
    loadsteps = [5,10,15,20,25,30]
elif 'GentThomas' in fem_material:
    loadsteps = [10,20,30]
elif 'ArrudaBoyce' in fem_material:
    loadsteps = [5,10,15,20,25,30,35,40,45,50]
else:
    loadsteps = [10,20,30,40,50,60,70,80]

#=====================================================================
# DATA:
#=====================================================================
for loadstep in loadsteps:

    data_path = get_data_path(fem_dir, fem_material,
                              noise_level, loadstep)
    data = loadFemData(data_path)
    datasets.append(data)

model = convexKAN(n_hidden=n_hidden,
                 grid_range=grid_range,
                 seed=0)


print_model_arch(model)

if(random_init):
    model.apply(init_weights)

print('\n\n====================================================================')
print('Beginning training\n')
print('Training an ensemble of models...')
for ensemble_iter in range(ensemble_size):
    model, loss_history = train_weak(model, datasets)
    os.makedirs(output_dir+'/'+fem_material+'/',exist_ok=True)
    torch.save(model.state_dict(), output_dir+'/'+fem_material+'/noise='+noise_level+'_'+str(ensemble_iter)+'.pth')
    exportList(output_dir+'/'+fem_material+'/','loss_history_noise='+noise_level+'_'+str(ensemble_iter),loss_history)
    model = convexKAN(n_hidden=n_hidden,
                grid_range=grid_range,
                seed=ensemble_iter+1)
print('\n\n=========================================================================================================================')
print('Evaluating and plotting ICNN on standard strain paths.')
evaluate_ickan(model, fem_material, noise_level, plot_quantities, output_dir)
print('Completed.')
print('\n\n=========================================================================================================================')
print('Symbolic form of best performing model:')
get_symbolic(model, fem_material, noise_level, output_dir)
print('=========================================================================================================================\n\n')