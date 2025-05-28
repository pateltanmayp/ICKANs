from core import *
from config import *


def train_weak(model, datasets):

	#loss history
	loss_history = []

	# optimizer
	if(opt_method == 'adam'):
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif(opt_method == 'lbfgs'):
		optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
	elif(opt_method == 'sgd'):
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	elif(opt_method == 'rmsprop'):
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	else:
		raise ValueError('Incorrect choice of optimizer')

	#scheduler
	if lr_schedule == 'multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,lr_milestones,lr_decay,last_epoch=-1)
	elif lr_schedule == 'cyclic':
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=cycle_momentum, step_size_up=step_size_up, step_size_down=step_size_down)

	
	print('--------------------------------------------------------------------------------------------------------')
	print('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |')
	print('--------------------------------------------------------------------------------------------------------')


	for epoch_iter in range(epochs):

		def closure():

			optimizer.zero_grad()

			loss = torch.tensor([0.])

			def computeLosses(data, model):

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
				F11 = data.F[:,0:1]
				F12 = data.F[:,1:2]
				F21 = data.F[:,2:3]
				F22 = data.F[:,3:4]

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

				P = P_NN #+ P_cor

				# compute internal forces on nodes
				f_int_nodes = torch.zeros(data.numNodes,dim)
				for a in range(num_nodes_per_element):
					for i in range(dim):
						for j in range(dim):
							force = P[:,voigt_map[i][j]] * data.gradNa[a][:,j] * data.qpWeights
							f_int_nodes[:,i].index_add_(0,data.connectivity[a],force)

				# clone f_int_nodes_clone
				f_int_nodes_clone = f_int_nodes.clone()
				# set force on Dirichlet BC nodes to zero
				f_int_nodes_clone[data.dirichlet_nodes] = 0.
				# loss for force equillibrium
				eqb_loss = torch.sum(f_int_nodes_clone**2)

				reaction_loss = torch.tensor([0.])
				for reaction in data.reactions:
					reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]) - reaction.force)**2

				return eqb_loss, reaction_loss

			for data in datasets:

				eqb_loss, reaction_loss = computeLosses(data, model)
				loss += eqb_loss_factor * eqb_loss + reaction_loss_factor * reaction_loss
				

			# back propagate
			loss.backward()
			
			return loss, eqb_loss, reaction_loss

		loss, eqb_loss, reaction_loss = optimizer.step(closure)

		scheduler.step()

		if(epoch_iter % verbose_frequency == 0):
			
			print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E |' % (
				epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item()))
			loss_history.append([epoch_iter+1,loss.item()])
	
	return model, loss_history
