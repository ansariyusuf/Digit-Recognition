from __future__ import division
from collections import OrderedDict
from commons import Variable
import numpy as np

dtype = np.float32


class DenseLayer(object):
    """A fully connected dense layer

    Parameters:
        w: in_dim x out_dim: The weight matrix
        n: out_dim, : the bias

    Arguments:
        n_out: number of output features
        init_type: the type of initialization
            can be gaussian or uniform

    """

    def __init__(self, n_out, init_type):
        self.n_out = n_out
        self.init_type = init_type
        self.params = OrderedDict()
        self.params["w"] = Variable()
        self.params["b"] = Variable()

    def get_output_dim(self):
        # The output dimension
        return self.n_out

    def init(self, n_in):
        # initializing the network, given input dimension
        scale = np.sqrt(1. / (n_in))
        if self.init_type == "gaussian":
            self.params["w"].value = scale * np.random.normal(
                0, 1, (n_in, self.n_out)).astype(dtype)
        elif self.init_type == "uniform":
            self.params["w"].value = 2 * scale * np.random.rand(
                n_in, self.n_out).astype(dtype) - scale
        else:
            raise NotImplementedError("{0} init type not found".format(
                self.init_type))
        self.params["b"].value = np.zeros((self.n_out), dtype=dtype)

    def forward(self, inputs):
        """The forward pass

        Arguments:
            inputs (``OrderedDict``): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array n_in dimensions

        Returns:
            outputs (``OrderedDict``): A dictionary containing
                height: The height of the output (1 for a dense layer)
                width: The width of the output (1 for a dense layer)
                out_channels: The output number of feature maps
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    n_out dimensions

        """
        data = inputs["data"]
        outputs = OrderedDict()
        # cache for backward pass
        self.data = data
        #print(data.shape,"input data shape dense")
        #print(data)
        outputs = OrderedDict()
        outputs["height"] = 1
        outputs["width"] = 1
        outputs["channels"] = self.n_out
        outputs["data"]=np.zeros(
            (data.shape[0], self.n_out), dtype=dtype)
        #print(self.n_out)
        weights=self.params["w"].value.transpose()
        offset=self.params["b"].value
        #print(weights.shape,"weight shape dense")
        #print(offset.shape,"offset shape dense")
        
        """for i in range(data.shape[0]):
            example=data[i]
            for j in range(weights.shape[0]):
                outputs["data"][i][j]=np.dot(example,weights[j])+offset[j]"""
                
        
        #print(val.shape,"shape of the output")
        #offset_matrix=np.array([offset,]*data.shape[0])
        # print(offset_matrix.shape,"shape of offset matrix")
        #final_output=np.add(val,offset_matrix)

        w=self.params["w"].value
        val=np.dot(data,w)
        outputs["data"]=np.add(val,offset)
        return outputs

    def backward(self, output_grads):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that this layer also computes the gradients wrt the
        parameters (i.e you should populate the values of
        self.params["w"].grad, and self.params["b"].grad here)

        Note that you should compute the average gradient
        (i.e divide by batch_size) when you computing the gradient
        of parameters.
        """
        grad = output_grads["grad"]
        #print(grad.shape,"output_grad shape")
        '''self.params["w"].grad=np.zeros(
            (self.params["w"].value.shape[1], self.params["w"].value.shape[0]), dtype=dtype)
        
        for i in range(self.params["w"].value.shape[1]):
            for j in range(self.params["w"].value.shape[0]):

                grad1=0
                for k in range(output_grads["grad"].shape[0]):
                    grad1+=output_grads["grad"][k][i]*(self.data[k][j])

                grad1=grad1/(self.data.shape[0])  #dividing by batch size
                self.params["w"].grad[i][j]=grad1
        self.params["w"].grad=self.params["w"].grad.transpose()'''
        #gradient for w
        self.params["w"].grad=np.dot(self.data.transpose(),grad)
        self.params["w"].grad=np.true_divide(self.params["w"].grad,self.data.shape[0])
        
        self.params["b"].grad=np.zeros(
            self.params["b"].value.shape, dtype=dtype)
        
        #Gradient for b
        temp=grad.transpose()
        temp=np.sum(temp, axis=1)
        self.params["b"].grad=np.true_divide(temp,self.data.shape[0])
        '''
        for i in range(self.params["b"].value.shape[0]):
            val=0
            for j in range(grad.shape[0]):
                val+=grad[j][i]
            self.params["b"].grad[i]=val/self.data.shape[0]'''

        #Gradient with respect to input
        input_grads = OrderedDict()
        input_grads["grad"]=np.dot(grad,self.params["w"].value.transpose())
        '''input_grads["grad"]=np.zeros(
            self.data.shape, dtype=dtype)

        weights=self.params["w"].value
        for ix in range(self.data.shape[0]):
            curr_example=self.data[ix]

            for i in range(self.data.shape[1]):
                val=np.dot(weights[i],grad[ix])
                input_grads["grad"][ix][i]=val'''
        
        return input_grads


class ReLULayer(object):
    """A ReLU activation layer
    """

    def __init__(self):
        self.params = OrderedDict()
        
    def my_func(self,a):
        if(a>0):
            return a
        else:
            return 0
	    
    def forward(self, inputs):
        """The forward pass

        Arguments:
            inputs (``OrderedDict``): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array n_in dimensions

        Returns:
            outputs (``OrderedDict``): A dictionary containing
                height: The height of the output (1 for a dense layer)
                width: The width of the output (1 for a dense layer)
                out_channels: The output number of feature maps
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    n_in

        Note that you only need to populate the outputs["data"]
        element.
        """
        outputs = OrderedDict()
        for key in inputs:
            if key != "data":
                outputs[key] = inputs[key]
            else:
                # hash for backward pass
                self.data = inputs[key]
                '''outputs["data"] = np.zeros(
                inputs["data"].shape, dtype=dtype)
                for i in range(inputs["data"].shape[0]):
                    for j in range(inputs["data"].shape[1]):
                        if(inputs["data"][i][j]>0):
                            outputs["data"][i][j]=inputs["data"][i][j]
                        else:
                            outputs["data"][i][j]=0'''

                outputs["data"]=np.clip(inputs["data"],a_min = 0,a_max = None)
                #func=np.vectorize(self.my_func,otypes=[np.float])
                #outputs["data"]=func(inputs["data"])
                
        '''print(outputs["data"],"output shape")6
        print(outputs["data"][a][b],"checking")'''
        return outputs

    def backward(self, outputs_grad):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that you just compute the gradient wrt the ReLU layer
        """
        inputs_grad = OrderedDict()
        temp=np.zeros(
            self.data.shape, dtype=dtype)

        condition=self.data>0
        inputs_grad["grad"]=np.where(condition,outputs_grad["grad"],temp)
        
        '''for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if(self.data[i][j]>0):
                    inputs_grad["grad"][i][j]=1
                else:
                    inputs_grad["grad"][i][j]=0'''
        return inputs_grad
