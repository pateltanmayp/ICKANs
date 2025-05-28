from .__importList__ import *

def computeFeatures(I1, I2, I3):
    """
    Compute the features dependent on the right Cauchy-Green strain invariants.
    Note that the features only depend on I1 and I3.

    _Input Arguments_

    - `I1` - 1st invariant

    - `I2` - 2nd invariant

    - `I3` - 3rd invariant

    _Output Arguments_

    - `x` - features

    ---

    """
    # numFeatures = 3
    # x = torch.zeros(I1.shape[0],numFeatures)

    K1 = I1 * torch.pow(I3,-1/3) - 3.0
    K2 = (I1 + I3 - 1) * torch.pow(I3,-2/3) - 3.0
    J = torch.sqrt(I3)
    K3 = (J-1)**2

    x = torch.cat((K1,K2,K3),1).float()
    return x

def getNumberOfFeatures():
    """
    Compute number of features.

    _Input Arguments_

    - _none_

    _Output Arguments_

    - `features.shape[1]` - number of features

    ---

    """
    features = computeFeatures(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1))
    return features.shape[1]


class Normalization:
    def __init__(self, datasets, flag=False):

        print('\n-----------------------------------------------------')
        print('Normalization info:')

        device = datasets[0].I1.device

        x = torch.tensor([]).cpu()
        for data in datasets:
            x = torch.cat((x, computeFeatures(data.I1,data.I2,data.I3).cpu()), dim=0)

        self.minVal = torch.min(x,dim=0,keepdim=False)[0].detach().to(device)
        self.maxVal = torch.max(x,dim=0,keepdim=False)[0].detach().to(device)
        self.meanVal = torch.mean(x,dim=0,keepdim=False).detach().to(device)
        self.stdVal = torch.std(x,dim=0,keepdim=False).detach().to(device)

        self.flag = flag
        self.columnFlag = (torch.abs(self.maxVal - self.minVal)>1e-7).cpu().squeeze().numpy().tolist()
        print('apply: ',self.flag)
        print(' flag: ',self.columnFlag)
        print('  min: ',self.minVal)
        print('  max: ',self.maxVal)
        print(' diff: ',self.maxVal-self.minVal)
        print('-----------------------------------------------------\n')

    def apply(self, x):
        y=x.clone()
        if(self.flag):
            for i in range(x.shape[1]):
                if(self.columnFlag[i]):
                    y[:,i] = (x[:,i]-self.minVal[i])/(self.maxVal[i]-self.minVal[i])
        return y
