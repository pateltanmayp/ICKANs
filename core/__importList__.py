#torch
import torch

#torch options
torch.manual_seed(123)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#numpy
import numpy as np
np.random.seed(123)

import pandas as pd
import os