#sys and core
import sys
sys.path.insert(0, "../")
from core import *
#config
import config as c
from ickan import *


class convexKAN(torch.nn.Module):
	def __init__(self, n_hidden, grid_range,seed=0):
		super(convexKAN, self).__init__()

		self.ICKAN = KAN(width=n_hidden, grid=c.grid, k=c.spline_order, seed=seed, device='cpu', base_fun='zero', grid_eps = 1.0, 
				   grid_range_0=grid_range, sp_trainable=c.sp_trainable, sb_trainable=c.sb_trainable,symbolic_enabled=c.symbolic_enabled,
				   auto_save=False)


	def forward(self, x):

		# Get F components
		F11 = x[:,0:1]
		F12 = x[:,1:2]
		F21 = x[:,2:3]
		F22 = x[:,3:4]

		# Compute right Cauchy green strain Tensor
		C11 = F11**2 + F21**2
		C12 = F11*F12 + F21*F22
		C21 = F11*F12 + F21*F22
		C22 = F12**2 + F22**2

		# Compute computeStrainInvariants
		I1 = C11 + C22 + 1.0
		I2 = C11 + C22 - C12*C21 + C11*C22
		I3 = C11*C22 - C12*C21

		# Apply transformation to invariants
		K1 = I1 * torch.pow(I3,-1/3) - 3.0
		K2 = torch.pow((I1 + I3 - 1.) * torch.pow(I3,-2./3.),3/2) - 3.0*torch.sqrt(torch.tensor(3.0))
		J = torch.sqrt(I3)
		K3 = (J-1)**2

		# Concatenate feature
		x_input = torch.cat((K1,K2,K3),1).float()

		y = self.ICKAN(x_input)

		return y

def init_weights(m):
	if isinstance(m, torch.nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight)

def print_model_arch(model):
	print('\n\n','-'*80,'\n',model,'\n','-'*80,'\n\n')


def print_model_params(model):
	print('\n\n====================================================================')
	print('Initial model:')
	print(model.state_dict())
	print('====================================================================\n\n')
