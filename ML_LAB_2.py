import scipy
import numpy as np
from numpy import *
from numpy.linalg import *
import matplotlib
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from functools import partial
import math
from pylab import imread, gray
from IPython.core.display import Image 


class Node(object):
    """
    Base-class for Nodes in a factor graph. Only instantiate sub-classes of Node.
    """
    def __init__(self, name):
        # A name for this Node, for printing purposes
        self.name = name
        
        # Neighbours in the graph, identified with their index in this list.
        # i.e. self.neighbours contains neighbour 0 through len(self.neighbours) - 1.
        self.neighbours = []
        
        # Reset the node-state (not the graph topology)
        self.reset()
        
    def reset(self):
        # Incomming messages; a dictionary mapping neighbours to messages.
        # That is, it maps  Node -> np.ndarray.
        self.in_msgs = {}
        
        # A set of neighbours for which this node has pending messages.
        # We use a python set object so we don't have to worry about duplicates.
        self.pending = set([])
        
        self.loopy = True

    def add_neighbour(self, nb):
        self.neighbours.append(nb)
        if self.loopy:
            if isinstance(nb, Factor):
                self.in_msgs[nb.name] = np.zeros(self.num_states)
            else:
                self.in_msgs[nb.name] = np.zeros(nb.num_states)
            self.pending.add(nb)

    def send_sp_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_sp_msg not implemented in base-class Node')
   
    def send_ms_msg(self, other):
        # To be implemented in subclass.
        raise Exception('Method send_ms_msg not implemented in base-class Node')
    
    def receive_msg(self, other, msg):
        ##################### Receive Incoming Message #################################################
        ## Store the incomming message, replacing previous messages from the same node.
        self.in_msgs[other.name] = msg
        
        ## Check which neighbours to put in the pending list . Here we get a list of the names of all the neighbours.
        message_keys = set(self.in_msgs.keys())
        nb_names = []
        for nb in self.neighbours:
            nb_names += [nb.name]
        
        ## Depending on wether we are using the loopy or non-loopy version of the algorithm , we need to 
        ## either put all remaining neighbours to pending , or just the ones that we can send a message to.
        if not self.loopy :
            for neighbor in self.neighbours:
                if neighbor.name==other.name: 
                    continue
                p = set(nb_names)
                p.remove(neighbor.name)
                if p.issubset(message_keys):
                    self.pending.add(neighbor)
        else:
            p = list(self.neighbours)
            p.remove(other)
            self.pending.update(p)
        ################################################################################################
       
    
    def __str__(self):
        # This is printed when using 'print node_instance'
        return self.name


class Variable(Node):
    def __init__(self, name, num_states):
        """
        Variable node constructor.
        Args:
            name: a name string for this node. Used for printing. 
            num_states: the number of states this variable can take.
            Allowable states run from 0 through (num_states - 1).
            For example, for a binary variable num_states=2,
            and the allowable states are 0, 1.
        """
        self.num_states = num_states
        
        # Call the base-class constructor
        super(Variable, self).__init__(name)
    
    def set_observed(self, observed_state):
        """
        Set this variable to an observed state.
        Args:
            observed_state: an integer value in [0, self.num_states - 1].
        """
        # Observed state is represented as a 1-of-N variable
        # Could be 0.0 for sum-product, but log(0.0) = -inf so a tiny value is preferable for max-sum
        self.observed_state[:] = 0.000001
        self.observed_state[observed_state] = 1.0
        
    def set_latent(self):
        """
        Erase an observed state for this variable and consider it latent again.
        """
        # No state is preferred, so set all entries of observed_state to 1.0
        # Using this representation we need not differentiate observed an latent
        # variables when sending messages.
        self.observed_state[:] = 1.0
        
    def reset(self):
        super(Variable, self).reset()
        self.observed_state = np.ones(self.num_states)
        
    def marginal(self, Z=None):
        ####################### Variable Marginal Distribution ##############################################
        ## Compute the marginal distribution of this Variable. It is assumed that message passing has completed
        ## when this function is called.
        ##    Args:
        ##         Z: an optional normalization constant can be passed in. If None is passed, Z is computed.
        ##    Returns: marginal, Z. The first is a numpy array containing the normalized marginal distribution.
        
        message = self.observed_state.copy()
        for m in self.in_msgs:
            for i in range(len(message)):
                message[i] *= self.in_msgs[m][i]
                
        ## Z is either equal to the input Z, or computed in this function (if Z=None was passed).
        if Z==None:  
            Z = sum(message) 
            
        message = message/Z
        return message , Z
        #####################################################################################################

    def map_state(self):
        ######################## Variable MAP state #########################################################
        ## Get the MAP state of the variable after running the max-sum algorithm .We assume that we have 
        ## already run the max-sum algorithm before we run this method.
        message = log(self.observed_state.copy())
        for m in self.in_msgs:
            #message += self.in_msgs[m]
            for i in range(len(message)):
                message[i] += self.in_msgs[m][i]
        return argmax(message) 
        #####################################################################################################
    
    def send_sp_msg(self, other):
        ######################### Sum-Product Message Sending ###############################################
        ## Message is initialised as the observed state (if not observed it is all 'ones'). After that we 
        ## multiply all incoming messages by element.
        message = self.observed_state.copy()
        for i in self.neighbours:
            if i.name != other.name:
                assert i.name in self.in_msgs
                for j in range(len(message)):
                    message[j] *= self.in_msgs[i.name][j]
        ## Finaly we send the message to the 'other' node by calling its 'receive_msg' method  , and remove
        ## 'other' from pending.
        ##print self.name +" -----  " , message , " -------> " + other.name
        other.receive_msg(self, message)
        self.pending.remove(other)
        #####################################################################################################
           
    def send_ms_msg(self, other):
        ######################### Max-Sum Message Sending ###################################################
        ## Message is now initialised as the ln of the observed state. After that we sum all incoming mesages
         
        message = log(self.observed_state.copy())
        for i in self.neighbours:
            if i.name != other.name:
                assert i.name in self.in_msgs
                message += self.in_msgs[i.name]
        ## Finaly we send the message to the 'other' node by calling its 'receive_msg' method , and remove
        ## 'other' from pending.
        ##print self.name +" -----  " , message , " -------> " + other.name
        other.receive_msg(self, message)
        self.pending.remove(other)
        #####################################################################################################

class Factor(Node):
    def __init__(self, name, f, neighbours):
        """
        Factor node constructor.
        Args:
            name: a name string for this node. Used for printing
            f: a numpy.ndarray with N axes, where N is the number of neighbours.
               That is, the axes of f correspond to variables, and the index along that axes corresponds to a value of that variable.
               Each axis of the array should have as many entries as the corresponding neighbour variable has states.
            neighbours: a list of neighbouring Variables. Bi-directional connections are created.
        """
        # Call the base-class constructor
        super(Factor, self).__init__(name)
        assert len(neighbours) == f.ndim, 'Factor function f should accept as many arguments as this Factor node has neighbours'
        
        for nb_ind in range(len(neighbours)):
            nb = neighbours[nb_ind]
            assert f.shape[nb_ind] == nb.num_states, 'The range of the factor function f is invalid for input %i %s' % (nb_ind, nb.name)
            self.add_neighbour(nb)
            nb.add_neighbour(self)

        self.f = f
        
    def send_sp_msg(self, other):
        ######################### Sum-Product Message Sending ###############################################
        ## To send a message to other we need to filter out all the incoming messages that come from 'other'
        ## and get the indexes of all the remaining neighbours.
        incoming_messages = []
        incoming_indexes = []
        for i in self.neighbours:

                assert i.name in self.in_msgs
                incoming_messages+= [self.in_msgs[i.name] ]
                incoming_indexes += [self.neighbours.index(i)]
                
        ## Then we create a tensor using all the incoming messages and using tensordot we marginalise out all
        ## the 'incoming' nodes (all nodes except the one we send to). 
        M = np.multiply.reduce(np.ix_(*incoming_messages)) 
        message = tensordot( self.f, M , (tuple(incoming_indexes) , tuple(range(len(incoming_indexes))) ) )
        
        ## Finaly we send the message to the 'other' node by calling its 'receive_msg' method , and remove
        ## 'other' from pending.
        ##print self.name +" -----  " , message , " -------> " + other.name
        other.receive_msg(self , message)
        self.pending.remove(other)
        #####################################################################################################
        
    def send_ms_msg(self, other):
        ######################### Max-Sum Message Sending ###################################################
        ## To send a message to other we need to filter out all the incoming messages that come from 'other'
        ## and get the indexes of all the remaining neighbours.
        incoming_messages = []
        incoming_indexes = []
        for i in self.neighbours:
            if i.name != other.name:
                assert i.name in self.in_msgs
                incoming_messages+= [self.in_msgs[i.name] ]
                incoming_indexes += [self.neighbours.index(i)]
                
        ## In the Max-Sum we add instead of multiplying , as we now use logarithm values.         
        M = np.add.reduce(np.ix_(*incoming_messages)) 
        BigMatrix = log(self.f) + M 
        
        ## Finaly we need to get the maximum values for every case of 'other' . To do this we need to find
        ## the index value of 'other' and transpose the big matrix accordingly. Then we run amax() for all the
        ## number of states of 'other'.
        index_other = self.neighbours.index(other)
        params =  [index_other]  + incoming_indexes
        b = transpose(BigMatrix, tuple(params))
        message = []
        for i in range(other.num_states):
            message += [amax(b[i])]
            
        ## Finaly we send the message to the 'other' node by calling its 'receive_msg' method , and remove
        ## 'other' from pending.
        ##print self.name +" -----  " , message , " -------> " + other.name
        other.receive_msg(self , message)
        self.pending.remove(other)
        #####################################################################################################









# Load the image and binarize
im = np.mean(imread('dalmatian1.png'), axis=2) > 0.5
noise = np.random.rand(*im.shape) > 0.9
noise_im = np.logical_xor(noise, im)
# test_im = np.zeros((200,200))
# test_im[50:150, 50:150] = 1.0
# test_im[5,5] = 1.0
# figure()
# imshow(test_im)

# # Add some noise
# noise = np.random.rand(*test_im.shape) > 0.8
# noise_test_im = np.logical_xor(noise, test_im)
# figure()
# imshow(noise_test_im)
# print noise_im.shape
# print noise_test_im


observed_variables = {}
latent_variables = {}
ol_factors = {}
l_factors = {}
lp_factors = {}
a = 0.7
h = 0.5
b = 0.9
# A = array([[exp(a) , exp(-a)],[exp(-a) , exp(a)]])
# H = array([ exp(h) , exp(-h)]) 
# B = array([[exp(b) , exp(-b)],[exp(-b) , exp(b)]])

A = array([[a , 1-a],[1-a , a]])
H = array([ h , 1-h]) 
B = array([[b , 1-b],[1-b , b]])

o = 0
l = 0
for i,row in enumerate(noise_im):
    for j,item in enumerate(row):
        observed_variables[i,j] = Variable('y'+str(o),2)
        latent_variables[i,j] = Variable('x'+str(o),2)
        ol_factors[i,j] = Factor('fxy'+str(o), A ,[ observed_variables[i,j] , latent_variables[i,j] ])
        lp_factors[i,j] = Factor('fx'+str(o), H  ,[latent_variables[i,j]])
        o+=1
        observed_variables[i,j].set_observed(item)
        ##latent_variables[i,j].set_observed(item)
        if i > 0:
            l_factors[i,j,i-1,j] = Factor ('fxx'+ str(l), B ,[latent_variables[i,j] , latent_variables[i-1,j]])
            l+=1
        if j > 0:
            l_factors[i,j,i,j-1] = Factor ('fxx'+ str(l), B ,[latent_variables[i,j] , latent_variables[i,j-1]])
            l+=1

nodes = observed_variables.values() + ol_factors.values() + lp_factors.values()  + latent_variables.values() + l_factors.values() 

def send_all_pending(node):
    pend = node.pending.copy()
    for other in pend:

        node.send_ms_msg(other)
print len(nodes)

import random
iterations = 100000
for i in range(iterations):
    if i%100000 == 0 : print i
    node = random.sample(nodes,1)
    send_all_pending(node[0])
#     print "----------------------------------------"
#     for group in nodes:
#         for node in group:
# #            print group[node].name
#             send_all_pending(group[node])
#             #print group[node].pending
#             #break



new_im = np.zeros((600,750))
for i,row in enumerate(noise_test_im):
    for j,item in enumerate(row):
        new_im[i][j] = latent_variables[i,j].map_state()


scipy.misc.imsave('new_im.png', new_im)