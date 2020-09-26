#Contain the code that crates the YOLO network
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 

from util import *
import cv2

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_
def parse_cfg(cfgfile):
    """
    Takes a configuration file
    Return a list of blocks. Each blocks describes a block in the neural network to be build.
    Block is represented as a dictionary in the list
    """

    file = open(cfgfile,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x)>0]
    lines = [x for x in lines if x[0]!='#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []
    for line in lines:
        if line[0] =='[':
            if len(block)!=0:
                blocks.append(block)
                block={}
            block['type'] = line[1:-1].rstrip()
        else:
            key,value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3  # it is used to keep track of number of filters in the layers on which the convolutional layer  is being applied and equal to 3 as the image has 3 filters corresponding to RGB channels.
    output_filters = []  #append the number of output filters of each block to the list 

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()

        #check the type of block
        #create a new module for the block
        #append the module to module list

        #if the layer is convolutional layer it add convolutional layer along with Batch Normalization layer and Activation Layer
        if block['type']=='convolutional':
            #get the info about the convolutional layer
            activation = block['activation']
            try:
                batch_normalize=int(block['batch_normalize'])
                bias = False 
            except:
                batch_normalize=0
                bias =True
            
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            padding = int(block['pad'])
            stride = int(block['stride'])

            if padding:
                pad = (kernel_size-1)//2
            else:
                pad = 0
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias=bias)
            module.add_module("conv_{0}".format(index),conv)

            #add the batch normalization layer to the convolutional layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            #add the leaky relu activation function 
            #it is either Linear or a Leaky RelU for YOLO
            if activation=='leaky':
                activn = nn.LeakyReLU(0.1,inplace=True)
                module.add_module("leaky_{0}".format(index),activn)
        
        #if it's an upsampling layer
        elif block["type"]=="upsample":
            stride = int(block['stride'])
            upsample = nn.Upsample(scale_factor=2,mode="bilinear")
            module.add_module('upsample_{0}'.format(index),upsample)

        #if the layer is route
        elif block['type']=='route':
            block['layers'] = block['layers'].split(',')
            #start of the route 
            start = int(block['layers'][0])

            #end of the route if exist
            try:
                end = int(block['layers'][1])
            except:
                end = 0
            
            if start>0:
                start = start-index
            if end>0:
                end = end-index
            
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)
            if end <0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index + start]

        elif block['type']=="shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index),shortcut)

        elif block['type']=='yolo':
            mask = block['mask'].split(',')
            mask = [int(i) for i in mask]

            anchors = block['anchors'].split(',')
            anchors = [int(anchor) for anchor in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index),detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info,module_list)

class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self,x,CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i,module in enumerate(modules):
            module_type = module['type']
            if module_type=='convolutional'  or module_type=='upsample':
                x = self.module_list[i](x)
            elif module_type =='route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0]>0:     #start is  =greater than 0
                    layers[0] = layers[0] - i #calculate the relative path
                
                if len(layers) == 1:
                    x = outputs[i + layers[0]] #i +layers[0] is the absolute index of layer
                else:
                    if layers[1] >0: 
                        layers[1] = layers[1] - i #calculate the relative path of the end layer

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1,map2),1)
            elif module_type=='shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]
            elif module_type =="yolo":
                anchors = self.module_list[i][0].anchors

                #get the input dimensoinos

                inp_dim = int(self.net_info['height'])

                #get the number of classes
                num_classes = int(module['classes'])

                #transform
                x = x.data
                x = predict_transform(x,inp_dim,anchors,num_classes,CUDA)
                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections,x),1)
            outputs[i] = x
        return detections
    def load_weights(self,weightfile):
        #open the weights file
        fp = open(weightfile,'r')

        #The first 5 values are header information
        # 1. Major Version number
        # 2. Minor Version number
        # 3. SubVersion number
        # 4,5. Images seen by the network (during training)

        header = np.fromfile(fp,dtype=np.int32,count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp,dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            #if module_type == convolutional then load weights
            #otherwise ignore

            if module_type=='convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize = 0
                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    #get the number of weights of batch norm layer

                    num_bn_biases = bn.bias.numel()

                    #load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr +=num_bn_biases
                    
                    bn_weight = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    #cast the loaded weights into dims of model weights
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weight = bn_weight.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weight)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:

                    #number of biases
                    num_biases = conv.bias.numel()

                    #load weights 
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases

                    #reshape the loaded weights accroding to the dims of the model weihts
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    #finally copy the data
                    conv.bias.data.copy_(conv_biases)
                
                #let us load the weight for the convolutional layer
                num_weights = conv.weight.numel()

                # do the same as above for weights
                conv_weight = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights

                conv_weight = conv_weight.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weight)