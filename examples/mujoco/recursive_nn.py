from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch.nn as nn
import numpy as np
import xmltodict
from dataclasses import dataclass
#from urdfpy import URDF
from dm_control import mjcf
import yaml
import pdb

#@dataclass
#class Node:
#    name: str = None
#    observation_dim : int = 0
#    output_dim: int = 0
#    node_function: nn.Module = nn.Identity
    

class Node:
    def __init__(self,
                 name: str = None,
                 observation_dim : int = 0,
                 output_dim: int = 0,
                 node_function: nn.Module = nn.Identity):
        self.name = name
        self.observation_dim = observation_dim
        self.output_dim = output_dim
        


#@dataclass
class Joint(Node):
    def __init__(self,
                 parent : Link ,
                 children: Link,
                 **kwargs):
        super(Joint, self).__init__(**kwargs)        
        self.parent = parent
        self.children = children
                 

#@dataclass
#class Link(Node):
#    def __init__ (self, 
#                  parent  : Joint,
#                  children: list[Joint],
#                  **kwargs):
#        super(Link, self).__init__(**kwargs)
#        self.parent = parent
#        self.children = children
#        self.node_function = node_function

#@dataclass
class Link(Node):
    def __init__ (self, 
                  parent  :Joint,
                  children: list[Joint],
                  **kwargs):
        super(Link, self).__init__(**kwargs)
        self.parent = parent
        self.children = children


#def build_from_urdf(self, urdf_file):
#    robot = URDF.load(urdf_file)
#    pass

def build_link_from_mjcf_element(mjcf_model : mjcf.Element,
                                 nn_def: dict,
                                 activation: nn.Module) -> Link:
    #pdb.set_trace()
    print("Processing ", mjcf_model.name)
    observation_dim = nn_def[('Link',mjcf_model.name)]['sensor']
    output_dim = nn_def[('Link', mjcf_model.name)]['out']
    if observation_dim > 0 and output_dim > 0:
        node_function = nn.Sequential(
            nn.Linear(observation_dim, output_dim),
            activation
        )
    else:
        node_function = nn.Identity

    root = Link(parent = None,
                children = [],
                name = mjcf_model.name, 
                observation_dim = observation_dim,
                output_dim=output_dim,
                node_function = node_function)

    children = mjcf_model.body    
    if len(children) > 0:
        for child in children:
            print("Child of ", root.name, " is ", child.name )
            joint_mjcf = child.joint[0]
            child_link = build_link_from_mjcf_element(child, nn_def, activation)
            #if child.name == 'bfoot':
            #    pdb.set_trace()
            joint = Joint(parent=root,
                          children=child_link,
                          name = joint_mjcf.name,
                          observation_dim = nn_def[('Joint', joint_mjcf.name)]['sensor'],
                          output_dim = nn_def[('Joint',joint_mjcf.name)]['out'],
                          node_function = nn.Identity)

            child_link.parent = joint
            root.children.append(joint)
            
    return root
        

def build_from_mjcf_file(mjcf_fname : str,
                         root_name : str,
                         nn_def_yaml : str,
                         activation: nn.Module) -> Link:
    
    nn_def = yaml.load(stream=open(nn_def_yaml, 'r'))
    print("nn_def is ", nn_def)
    state_list = nn_def["state"]

    mjcf_model = mjcf.from_path(mjcf_fname)
    mjcf_root = mjcf_model.find('body', root_name)

    root = build_link_from_mjcf_element(mjcf_root,  nn_def, activation)
    return root, state_list, nn_def

class RecursiveNN(nn.Module):
    def __init__(self,
                 #state_shape: Union[int, Sequence[int]],
                 action_shape: Union[int, Sequence[int]],                 
                 mjcf_fname: str,
                 nn_def_yaml: str,
                 root_name: str,
                 #norm_layer: Optional[ModuleType] = None,
                 activation: Optional[ModuleType] = nn.ReLU(inplace=True),                 
                 device: Union[str, int, torch.device] = "cpu"
                 ):
        super(RecursiveNN, self).__init__()
        self.device = device
        self.root_name = root_name
        self.robot, self.state_list, self.nn_def = build_from_mjcf_file(mjcf_fname,
                                                                        root_name,
                                                                        nn_def_yaml,
                                                                        activation)
        #device)
        #self.softmax = softmax
        #self.num_atoms = num_atoms
        #self.input_dim = int(np.prod(state_shape))
        #self.action_dim = int(np.prod(action_shape)) #* num_atoms
        #if concat:
        #    input_dim += action_dim
        #self.use_dueling = dueling_param is not None
        #output_dim = action_dim if not self.use_dueling and not concat else 0
        #self.model = MLP(
        #    input_dim, output_dim, hidden_sizes, norm_layer, activation, device
        #)
        #self.output_dim = self.model.output_dim
        self.output_dim = action_shape

        self.neck = nn.Sequential(
            nn.Linear(self.robot.output_dim, self.robot.output_dim),
            activation,
            nn.Linear(self.robot.output_dim, self.robot.output_dim),
            activation,
            nn.Linear(self.robot.output_dim, self.output_dim)
        )
        
    def forward_curr_link(self,link : Link, x : dict) -> torch.Tensor:

        feat_curr_link = []

        # from current link's state observations
        if (link.name == self.root_name and link.observation_dim > 0):
            feat_curr_link.append( x[('Link', link.name)] )

        # from all joints'
        for joint in link.children:
            if joint.observation_dim > 0:
                joint_feat = x[('Joint', joint.name)]
                feat_curr_link.append(joint_feat)
            if joint.children is not None and joint.children.observation_dim > 0:
                feat_curr_link.append( self.forward_curr_link(joint.children, x))
        pdb.set_trace()
        return link.node_function(torch.cat(feat_curr_link, dim=1))

    def obs_to_dict(self,
                    obs : Union[np.ndarray, torch.Tensor] ):
        obs_num = obs.shape[-1]
        dict_obs = {}
        root_dim = self.robot.output_dim
        if len(obs.shape) > 1:
            dict_obs[('Link' ,self.root_name)] = np.zeros((obs.shape[0], self.robot.observation_dim))
        else:
            dict_obs[('Link',self.root_name)] = np.zeros((self.robot.observation_dim))

        root_idx = 0
        if len(obs.shape) > 1:
            for i in range(obs_num):
                pdb.set_trace()
                if self.state_list[i] in dict_obs:
                    dict_obs[self.state_list[i]][:, root_idx] =  obs[:, i]
                else:
                    dict_obs[self.state_list[i]] = obs[:, i]
                    root_idx += 1
        else:
            for i in range(obs_num):
                if self.state_list[i] in dict_obs:
                    dict_obs[self.state_list[i]][root_idx] =  obs[i]
                else:
                    dict_obs[self.state_list[i]] = obs[ i]
                    root_idx += 1
        return dict_obs
    
    def forward(self, #x : dict,
                obs: Union[np.ndarray, torch.Tensor],
                state: Any = None,
                info: Dict[str, Any] = {}
                ):
        x = self.obs_to_dict(obs)
        features = self.forward_curr_link(self.robot, x)
        logits = self.neck(features)
        return logits, state
        
    
if __name__ == '__main__':

    robot_nn = RecursiveNN(6, "half_cheetah.xml",
                           "half_cheetah.yaml", "torso"
                           )
    print("Just imported robot_nn")
    data = np.zeros((2,17), dtype=np.float)
    for i in range(17):
        data[:, i] = float(i)
    logits, _ = robot_nn(data)
    print(logits)
    
