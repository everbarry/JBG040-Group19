import torch.nn as nn
from torch import rand

from torchviz import make_dot
import graphviz

import argparse
from typing import List, Any


def modelvis(model: nn.Module, argv: argparse.Namespace, device:str = "cpu") -> graphviz.Graph:
    """
    Create a dot graphvis object that is the graphical representation of the network

    Inputs:
    model : pytorch network
    argv  : argparsed class arguments
    device: device the model is on


    Output:
    Graph object meant for further visualization

    """
    X = rand((1,1,128,128)).to(device)
    y = model(X)

    saved    = getattr(argv, 'showsaved'   , False)    
    showattr = getattr(argv, 'showattrib'  , False)   
    silent   = getattr(argv, 'viewmodelviz', False) 

    dot = make_dot(y.mean(),
                   params=dict(model.named_parameters()),
                   show_saved=saved,
                   show_attrs=showattr)

    # show in browser
    if not silent:
        dot.view()

    return dot



