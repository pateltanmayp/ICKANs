import sys
import matplotlib
sys.path.insert(0, '../')
from core import *
#config
from config import *
import config as c
#CUDA
initCUDA(cuda)
#supporting files
from model import *
from helper import *
from matplotlib.ticker import FormatStrFormatter

matplotlib.pyplot.rcParams['font.family'] = 'serif'
matplotlib.pyplot.rcParams['mathtext.fontset'] = 'dejavuserif'


def evaluate_ickan(model, fem_material, noise_level, plot_quantities, output_dir):

    """
    Evaluates the trained model along six deformation paths and compares it to the ground truth model.

    Arguments:

    model:				Trained model class instance.
    fem_material:		String specifying the name of the hidden material
    noise_level:		{low,high}
    plot_quantities:	Which quantites to plot {W,P}
    output_dir:			Output directory name

    """


    model.eval()

    def compute_corrected_W(F):
        """
        Compute the strain energy density according to Ansatz (Eq. 8).

        Input: 		Deformation gradient F in form (F11,F12,F21,F22)
        Output: 	Strain energy density according to Ansatz (Eq. 8)

        """

        F_0 = torch.zeros((1,4))
        F_0[:,0] = 1
        F_0[:,3] = 1

        F11_0 = F_0[:,0:1]
        F12_0 = F_0[:,1:2]
        F21_0 = F_0[:,2:3]
        F22_0 = F_0[:,3:4]

        F11_0.requires_grad = True
        F12_0.requires_grad = True
        F21_0.requires_grad = True
        F22_0.requires_grad = True

        # Get components of F
        F11 = F[:,0:1]
        F12 = F[:,1:2]
        F21 = F[:,2:3]
        F22 = F[:,3:4]

        F11.requires_grad = True
        F12.requires_grad = True
        F21.requires_grad = True
        F22.requires_grad = True

        W_NN = model(torch.cat((F11,F12,F21,F22),dim=1))

        dW_NN_dF11 = torch.autograd.grad(W_NN,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
        dW_NN_dF12 = torch.autograd.grad(W_NN,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
        dW_NN_dF21 = torch.autograd.grad(W_NN,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
        dW_NN_dF22 = torch.autograd.grad(W_NN,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

        P_NN = torch.cat((dW_NN_dF11,dW_NN_dF12,dW_NN_dF21,dW_NN_dF22),dim=1)

        # Derivative of volumetric term
        J_F_inv_T = torch.cat((F22,-F21,-F12,F11),1)
        C = computeCauchyGreenStrain(torch.cat((F11,F12,F21,F22),dim=1))
        _,_,I3 = computeStrainInvariants(C)

        # Zero deformation input data
        W_NN_0 = model(torch.cat((F11_0,F12_0,F21_0,F22_0),dim=1))

        dW_NN_dF11_0 = torch.autograd.grad(W_NN_0,F11_0,torch.ones(F11_0.shape[0],1),create_graph=True)[0]
        dW_NN_dF12_0 = torch.autograd.grad(W_NN_0,F12_0,torch.ones(F12_0.shape[0],1),create_graph=True)[0]
        dW_NN_dF21_0 = torch.autograd.grad(W_NN_0,F21_0,torch.ones(F21_0.shape[0],1),create_graph=True)[0]
        dW_NN_dF22_0 = torch.autograd.grad(W_NN_0,F22_0,torch.ones(F22_0.shape[0],1),create_graph=True)[0]

        P_NN_0 = torch.cat((dW_NN_dF11_0,dW_NN_dF12_0,dW_NN_dF21_0,dW_NN_dF22_0),dim=1)

        P_cor = torch.zeros_like(P_NN)

        P_cor[:,0:1] = F11*-P_NN_0[:,0:1] + F12*-P_NN_0[:,2:3]
        P_cor[:,1:2] = F11*-P_NN_0[:,1:2] + F12*-P_NN_0[:,3:4]
        P_cor[:,2:3] = F21*-P_NN_0[:,0:1] + F22*-P_NN_0[:,2:3]
        P_cor[:,3:4] = F21*-P_NN_0[:,1:2] + F22*-P_NN_0[:,3:4]

        P = P_NN + P_cor
        E = torch.zeros_like(F)

        E[:,0:1] = 0.5*(C[:,0:1]-1.)
        E[:,1:2] = 0.5*(C[:,1:2])
        E[:,2:3] = 0.5*(C[:,2:3])
        E[:,3:4] = 0.5*(C[:,3:4]-1.)

        W_cor = torch.sum(-P_NN_0*E, 1, keepdim=True)

        # The actual offset of the NN output is W_NN_0 (NN ouput at F=I) and H:E
        W_offset = W_NN_0
        # W is sum of NN output, the volumetric correction term and subtracting NN output at F=I and H:E
        W = W_NN + W_cor - W_offset

        P = P.view(-1,4,1)

        return W, P

    def get_true_W(fem_material,J,C,I1,I2,I3):
        """

        Analytical description of benchmark hyperelastic material models.
        Input:		Strain invariants and Cauchy-Green deformation matrix.
        Output:		Strain energy density of the specified material

        """
		
        I1_tilde = J**(-2/3)*I1

        x1 = (I1_tilde-3)
        x2 = torch.pow((I1 + I3 - 1.) * torch.pow(I3,-2./3.),3/2) - 3.0*torch.sqrt(torch.tensor(3.0))
        x3 = (J-1)**2

        if fem_material == 'NeoHookean':
            I1_tilde = J**(-2/3)*I1
            W_truth = 0.5000*(I1_tilde - 3) + 1.5000*(J - 1)**2
            if noise_level == 'high':
                W_symbolic = 0.53947*x1 + 1.48685*x3
            elif noise_level == 'low':
                W_symbolic = 0.52007*x1 + 1.49694*x3
				
        elif fem_material == 'Isihara':
            I1_tilde = J**(-2/3)*I1
            I2_tilde = J**(-4/3)*I2
            W_truth = 0.5000*(I1_tilde - 3) + 1.0000*(I2_tilde - 3) + 1.0000*(I1_tilde - 3)**2 + 1.5000*(J - 1)**2
			
            if noise_level == 'high':
                W_symbolic_0 = 0.00973*torch.log(10.16184*torch.exp(2.502*torch.tensor([0.])) + 1)**3 + 0.0039*torch.log(3568.85466*torch.exp(4.3*torch.tensor([0.])) + 1)**2 + 26.28099
                W_symbolic = 0.31363*x1 + 1.48415*x3 + 0.00973*torch.log(10.16184*torch.exp(2.502*x1) + 1)**3 + 0.0039*torch.log(3568.85466*torch.exp(4.3*x2) + 1)**2 + 26.28099 - W_symbolic_0
            elif noise_level == 'low':
                W_symbolic_0 = 0.004*torch.log(22.43181*torch.exp(3.304*torch.tensor([0.])) + 1)**3 + 0.00386*torch.log(3568.85466*torch.exp(4.29898*torch.tensor([0.])) + 1)**2 + 26.51481
                W_symbolic =  0.34624*x1 + 1.48514*x3 + 0.004*torch.log(22.43181*torch.exp(3.304*x1) + 1)**3 + 0.00386*torch.log(3568.85466*torch.exp(4.29898*x2) + 1)**2 + 26.51481 - W_symbolic_0
			
        elif fem_material == 'HainesWilson':
            I1_tilde = J**(-2/3)*I1
            I2_tilde = J**(-4/3)*I2
            W_truth = 0.5000*(I1_tilde - 3) + 1.0000*(I2_tilde - 3) + 0.7000*(I1_tilde - 3)*(I2_tilde - 3) + 0.2000*(I1_tilde - 3)**3 + 1.5000*(J - 1)**2
			
            if noise_level == 'high':
                W_symbolic_0 = 0.87824*torch.log(0.57194*torch.exp(1.03596*torch.tensor([0.])) + 1)**2 + 0.59131*torch.log(0.57267*torch.exp(1.038*torch.tensor([0.])) + 1)**2 + 24.10533
                W_symbolic = 0.41664*x2 + 1.49149*x3 + 0.87824*torch.log(0.57194*torch.exp(1.03596*x1) + 1)**2 + 0.59131*torch.log(0.57267*torch.exp(1.038*x1) + 1)**2 + 24.10533 - W_symbolic_0
            elif noise_level == 'low':
                W_symbolic_0 =  0.69473*torch.log(0.57267*torch.exp(1.038*torch.tensor([0.])) + 1)**2 + 0.00772*torch.log(6.64377*torch.exp(1.896*torch.tensor([0.])) + 1)**3 + 23.85097
                W_symbolic =  0.43086*x2 + 1.48226*x3 + 0.69473*torch.log(0.57267*torch.exp(1.038*x1) + 1)**2 + 0.00772*torch.log(6.64377*torch.exp(1.896*x1) + 1)**3 + 23.85097 - W_symbolic_0


        elif fem_material == 'GentThomas':
            I1_tilde = J**(-2/3)*I1
            I2_tilde = J**(-4/3)*I2
            W_truth = 0.5000*(I1_tilde - 3) + 1.5000*(J - 1)**2 + 1.0000*torch.log(I2_tilde/3)
            if noise_level == 'high':
                W_symbolic_0 = 1.09417*torch.log(16.44596*torch.exp(torch.tensor([0.])) + 1) 
                W_symbolic =  0.6237*x1 + 0.04369*x2 + 0.77956*x3 + 1.09417*torch.log(16.44596*torch.exp(0.718*x3) + 1) - W_symbolic_0
            elif noise_level == 'low':
                W_symbolic_0 = 1.57821*torch.log(35.20542*torch.exp(torch.tensor([0.]))+1)
                W_symbolic =  0.65074*x1 + 0.03226*x2 + 0.44194*x3 + 1.57821*torch.log(35.20542*torch.exp(0.69524*x3)+1) - W_symbolic_0
        
        elif fem_material == 'ArrudaBoyce':
            shear_modulus = 2.5
            chain_length = 28
            kappa = 1.5
            shearModulus = torch.tensor([shear_modulus])
            N = torch.tensor([chain_length])
            sqrt_N = torch.sqrt(N)
            W_truth = torch.zeros((gamma_steps,1))
            I1_tilde = J**(-2/3)*I1
            lambda_chain = 1
            x = lambda_chain / torch.sqrt(N)
            beta_chain = 0.0
            if torch.abs(x)<0.841:
                beta_chain = 1.31*torch.tan(1.59*x)+0.91*x
            else:
                beta_chain = 1./((x+0.00000001/(torch.abs(x)+0.00000001))-x)
            W_truth_offset = shearModulus * sqrt_N * (beta_chain*lambda_chain+sqrt_N*torch.log(beta_chain/torch.sinh(beta_chain)))
            for step in range(gamma_steps):
                lambda_chain = torch.sqrt(I1_tilde[step:step+1]/3.)
                x = lambda_chain / torch.sqrt(N)
                beta_chain = 0.0
                if torch.abs(x)<0.841:
                    beta_chain = 1.31*torch.tan(1.59*x)+0.91*x
                else:
                    beta_chain = 1./((x+0.00000001/(torch.abs(x)+0.00000001))-x)
                #% Final output
                W_truth[step:step+1,:] = shearModulus * sqrt_N * (beta_chain*lambda_chain+sqrt_N*torch.log(beta_chain/torch.sinh(beta_chain))) - W_truth_offset + kappa*(J[step:step+1]-1)**2
				
            if noise_level == 'high':
                W_symbolic_0 = 25.49315
                W_symbolic = 1.08613*x1 + 0.11399*x2 + 1.47873*x3 + 25.49315 - W_symbolic_0 

            elif noise_level == 'low':
                W_symbolic_0 = 25.50376
                W_symbolic = 1.19585*x1 +0.05333*x2 +1.4905*x3 + 25.50376 - W_symbolic_0 
            

        elif fem_material == 'Ogden':
            kappa_ogden = 1.5
            mu_ogden = 0.65
            alpha_ogden = 0.65
            I1_tilde = J**(-2/3)*I1 + 1e-13
            I1t_0 =torch.tensor([3]) + 1e-13
            J_0 = torch.tensor([1]) + 1e-13
            W_offset = kappa_ogden*(J_0-1)**2 + 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
            #% Final output
            W_truth = kappa_ogden*(J-1)**2 + 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset

            if noise_level == 'high':
                W_symbolic_0 = 0.22743*torch.log(17.64407*torch.exp(torch.tensor([0.])) + 1)
                W_symbolic =  0.69496*x1 + 0.042*x2 + 1.34817*x3 + 0.22743*torch.log(17.64407*torch.exp(0.7294*x3) + 1) - W_symbolic_0
            elif noise_level == 'low':
                W_symbolic_0 = 1.08851*torch.log(13.4637*torch.exp(torch.tensor([0.]))+1)
                W_symbolic =  0.75883*x1 + 0.00891*x2 + 0.8401*x3 + 1.08851*torch.log(13.4637*torch.exp(0.662*x3)+1) - W_symbolic_0


        return W_truth, W_symbolic

    for plot_quantity in plot_quantities:

        fig, axs = matplotlib.pyplot.subplots(3,3)
        fig.set_figwidth(9)
        fig.set_figheight(9)

        for i in range(3):
            fig.delaxes(axs[i,-1])

        axs_i = 0
        axs_j = 0

        for path_count, strain_path in enumerate(strain_paths):
            if strain_path == 'UT':
                P_idx = 0
            elif strain_path == 'SS':
                P_idx = 1
            elif strain_path == 'PS':
                P_idx = 3
            elif strain_path == 'UC':
                P_idx = 0
            elif strain_path == 'BT':
                P_idx = 0
            elif strain_path == 'BC':
                P_idx = 0

            print('Evaluating and plotting: '+fem_material+' for strain path '+strain_path)

            if 'SS' in strain_path or 'PS' in strain_path or 'UC' in strain_path:
                g_max = 1.0
            else:
                g_max = c.g_max

            gamma=np.linspace(c.g_min,g_max,gamma_steps)

            F, xlabel = getStrainPathDeformationGradient(strain_path, gamma_steps, gamma)

            # Get components of F
            F11 = F[:,0:1]
            F12 = F[:,1:2]
            F21 = F[:,2:3]
            F22 = F[:,3:4]

            F11.requires_grad = True
            F12.requires_grad = True
            F21.requires_grad = True
            F22.requires_grad = True

            #computing detF
            J = computeJacobian(torch.cat((F11,F12,F21,F22),dim=1))

            #computing Cauchy-Green strain: C = F^T F
            C = computeCauchyGreenStrain(torch.cat((F11,F12,F21,F22),dim=1))

            #computing strain invariants
            I1, I2, I3 = computeStrainInvariants(C)

            #Get true model of fem_material
            W_truth, W_symbolic = get_true_W(fem_material,J,C,I1,I2,I3)

            dW_truth_dF11 = torch.autograd.grad(W_truth,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
            dW_truth_dF12 = torch.autograd.grad(W_truth,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
            dW_truth_dF21 = torch.autograd.grad(W_truth,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
            dW_truth_dF22 = torch.autograd.grad(W_truth,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

            P_truth = torch.cat((dW_truth_dF11,dW_truth_dF12,dW_truth_dF21,dW_truth_dF22),dim=1)

            dW_sym_dF11 = torch.autograd.grad(W_symbolic,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
            dW_sym_dF12 = torch.autograd.grad(W_symbolic,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
            dW_sym_dF21 = torch.autograd.grad(W_symbolic,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
            dW_sym_dF22 = torch.autograd.grad(W_symbolic,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

            P_symbolic = torch.cat((dW_sym_dF11,dW_sym_dF12,dW_sym_dF21,dW_sym_dF22),dim=1)

            # Define plot bounds
            y_max_P = (torch.max(P_truth[:,P_idx])*1.1).detach().numpy()
            if y_max_P <= 0.:
                y_min_P = (torch.min(P_truth[:,P_idx])*1.1).detach().numpy()
                y_max_P = 0.0
            else:
                y_min_P = 0.

            final_losses = torch.zeros((ensemble_size,1))
            for ensemble_iter in range(ensemble_size):   
                final_losses[ensemble_iter] = pd.read_csv('results/'+fem_material+'/loss_history_noise='+noise_level+'_'+str(ensemble_iter)+'.csv', header=None).values[-1][1]
            
            idx_best_models = torch.topk(-final_losses.flatten(),1).indices

            W_prediction = torch.zeros((gamma_steps,1))
            P_prediction = torch.zeros((gamma_steps,4,1))

            model.load_state_dict(torch.load(output_dir+'/'+fem_material+'/noise='+noise_level+'_'+str(idx_best_models.item())+'.pth'))
            W_prediction, P_prediction = compute_corrected_W(F)


            if path_count < 3:
                axs_i = path_count
                axs_j = 0
            else:
                axs_i = path_count-3
                axs_j = 1

            if plot_quantity == 'P':
                if path_count == 0:
                    best, = axs[axs_i,axs_j].plot(gamma,P_prediction[:,P_idx,0].detach().numpy(), color=color_best, linestyle='--', 
                                                      marker = 'o', markevery=20, markerfacecolor='none', label='ICKAN', lw=lw_best, alpha=alpha_best, zorder=150)
                else:
                    axs[axs_i,axs_j].plot(gamma,P_prediction[:,P_idx].detach().numpy(), color=color_best, linestyle='--', 
                                          marker = 'o', markevery=20, markerfacecolor='none', lw=lw_best, alpha=alpha_best, zorder=150)


                if path_count == 0:
                    truth, = axs[axs_i,axs_j].plot(gamma,P_truth[:,P_idx].detach().numpy(),color=color_truth, lw=lw_truth, zorder=50)
                    symbolic, = axs[axs_i,axs_j].plot(gamma,P_symbolic[:,P_idx].detach().numpy(),color=color_sym, lw=lw_sym, alpha=alpha_sym,
                                                      linestyle='--', marker = '*', markevery=20, markerfacecolor='none', zorder=0, label='Symbolic')
                else:
                    axs[axs_i,axs_j].plot(gamma,P_truth[:,P_idx].detach().numpy(),color=color_truth,linestyle='-', lw=lw_truth)
                    axs[axs_i,axs_j].plot(gamma,P_symbolic[:,P_idx].detach().numpy(),color=color_sym, lw=lw_sym, alpha=alpha_sym,
                                                      linestyle='--', marker = '*', markevery=20, markerfacecolor='none', zorder=0)

                if P_idx == 0:
                    axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$P_{11}(\gamma)$',fontsize=fs)
                elif P_idx == 1:
                    axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$P_{12}(\gamma)$',fontsize=fs)
                elif P_idx == 3:
                    axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$P_{22}(\gamma)$',fontsize=fs)

                axs[axs_i,axs_j].set_ylim([y_min_P,y_max_P])
                axs[axs_i,axs_j].set_aspect(g_max/(y_max_P-y_min_P))
                axs[axs_i,axs_j].set_yticks([y_min_P,(y_max_P+y_min_P)/2,y_max_P])

                axs[axs_i,axs_j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axs[axs_i,axs_j].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axs[axs_i,axs_j].set_xlim([g_min,g_max])
                if path_count == len(strain_paths)-1:
                    axs[axs_i,axs_j].set_xlabel(r'$\gamma$',fontsize=fs)

                axs[axs_i,axs_j].set_xticks([0,g_max/2,g_max])
                axs[axs_i,axs_j].tick_params(axis='both', labelsize=fs)

            elif plot_quantity == 'W':

                y_min_W = (torch.min(W_truth)).detach().numpy()
                y_max_W = (torch.max(W_truth)*1.1).detach().numpy()

                if path_count == 0:
                    best, = axs[axs_i,axs_j].plot(gamma,W_prediction[:,0].detach().numpy(), label='ICKAN', color=color_best, linestyle='--',
                                                       marker = 'o', markevery=20, markerfacecolor='none', lw=lw_best, alpha=alpha_best, zorder=150)
                else:
                    axs[axs_i,axs_j].plot(gamma,W_prediction[:,0].detach().numpy(), color=color_best, linestyle='--',
                                                       marker = 'o', markevery=20, markerfacecolor='none', lw=lw_best, alpha=alpha_best, zorder=150)

                if path_count == 0:
                    truth, = axs[axs_i,axs_j].plot(gamma,W_truth[:,:].detach().numpy(),color=color_truth,label='True', lw=lw_truth, zorder=50)
                    symbolic, = axs[axs_i,axs_j].plot(gamma,W_symbolic[:].detach().numpy(),alpha=alpha_sym,color=color_sym, lw=lw_sym, linestyle='--',
                             marker = '*', markevery=20, markerfacecolor='none', zorder=0, label='Symbolic')
                else:
                    truth, = axs[axs_i,axs_j].plot(gamma,W_truth[:,:].detach().numpy(),color=color_truth,label='True', lw=lw_truth, zorder=50)
                    symbolic, = axs[axs_i,axs_j].plot(gamma,W_symbolic[:].detach().numpy(),alpha=alpha_sym,color=color_sym, lw=lw_sym, linestyle='--',
                             marker = '*', markevery=20, markerfacecolor='none', zorder=0)

                axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$W(\mathbf{F}(\gamma))$',fontsize=fs)

                axs[axs_i,axs_j].set_ylim([y_min_W,y_max_W])
                axs[axs_i,axs_j].set_aspect(g_max/(y_max_W-y_min_W))
                axs[axs_i,axs_j].set_yticks([y_min_W,(y_max_W+y_min_W)/2,y_max_W])

                axs[axs_i,axs_j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axs[axs_i,axs_j].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                axs[axs_i,axs_j].set_xlim([g_min,g_max])
                if path_count == len(strain_paths)-1:
                    axs[axs_i,axs_j].set_xlabel(r'$\gamma$',fontsize=fs)

                axs[axs_i,axs_j].set_xticks([0,g_max/2,g_max])
                axs[axs_i,axs_j].tick_params(axis='both', labelsize=fs)

            lgd = fig.legend(handles=[truth,best, symbolic],labels=['True','Best','Symbolic'],loc='upper right', bbox_to_anchor=(0.9, 0.9),fontsize=fs)

           
            suptitle = fig.suptitle(fem_material,fontsize=15)
            fig.tight_layout()

            if plot_quantity == 'P':
                matplotlib.pyplot.savefig(output_dir+'/benchmark='+fem_material+'_noise='+noise_level+'_Pij_.pdf',transparent=True,bbox_extra_artists=(lgd,suptitle,), bbox_inches='tight')
            elif plot_quantity == 'W':
                matplotlib.pyplot.savefig(output_dir+'/benchmark='+fem_material+'_noise='+noise_level+'_W.pdf',transparent=True,bbox_extra_artists=(lgd,suptitle,), bbox_inches='tight')


def get_symbolic(model, fem_material, noise_level, output_dir):

    
    loadstep = 50 # Changing loadstep (50) to a different number may change result.

    data_path = get_data_path(fem_dir, fem_material,
                                noise_level, loadstep) 
    data = loadFemData(data_path)

    model.ICKAN.get_act(data.F)
    model.ICKAN.plot()

    losses = []
    for i in range(ensemble_size):
        losses.append(pd.read_csv(output_dir+fem_material+'/loss_history_noise='+noise_level+'_'+str(i)+'.csv', header=None).values[-1,1])

    best_idx = np.argmin(losses)
    model.load_state_dict(torch.load(output_dir+fem_material+'/noise='+noise_level+'_'+str(best_idx)+'.pth'))

    
    model.ICKAN.auto_symbolic(lib=lib, verbose=True, a_range=(0,10), weight_simple=0.8)
    from ickan.utils import ex_round
    print(ex_round(model.ICKAN.symbolic_formula()[0][0],5))