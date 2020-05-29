from __future__ import division
from conv_layer import BaseConvLayer
from collections import OrderedDict
import numpy as np

dtype = np.float32


class PoolLayer(BaseConvLayer):
    def __init__(self, act_type, kernel_size, stride, pad):
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

        self.params = OrderedDict()
        super(PoolLayer, self).__init__()

    def init(self, height, width, in_channels):
        """No need to implement this func"""
        pass

    def forward(self, inputs):
        """The forward pass

        Arguments:
            inputs (OrderedDict): A dictionary containing
                height: the height of the current input
                width: the width of the current input
                channels: number of channels in the current
                    inputs
                data: a flattened data array of the form
                    batch_size * height * width * channel, unrolled in
                    the same way
        Returns:
            outputs (OrderedDict): A dictionary containing
                height: The height of the output
                width: The width of the output
                out_channels: The output number of feature maps
                    Same as the input channels for this layer
                data: a flattened output data array of the form
                    height * width * channel, unrolled in
                    the same way

        You may want to take a look at the im2col_conv and col2im_conv
        functions present in the base class ``BaseConvLayer``

        You may also find it useful to cache the height, width and
        channel of the input image for the backward pass.
        The output heights, widths and channels can be computed
        on the fly using the ``get_output_dim`` function.

        """
        h_in = inputs["height"]
        w_in = inputs["width"]
        c = inputs["channels"]
        data = inputs["data"]
        batch_size = data.shape[0]
        k = self.kernel_size
        h_out, w_out, c = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, c
        )
        # cache for backward pass
        self.h_in, self.w_in, self.c = h_in, w_in, c
        self.data = data

        outputs = OrderedDict()
        outputs["height"] = h_out
        outputs["width"] = w_out
        outputs["channels"] = c

        outputs["data"] = np.zeros(
            (batch_size, h_out * w_out * c), dtype=dtype)

        for ix in range(batch_size):
             col = self.im2col_conv(data[ix], h_in, w_in, c, h_out, w_out)
             '''col=col.reshape(k*k,h_out*w_out*c).transpose()
             outputs["data"][ix]=np.amax(col,axis=1)'''
             col=col.reshape(k*k,c,h_out*w_out)
             col=col.transpose(0,2,1)
             col=col.reshape(k*k,h_out*w_out*c).transpose()
             outputs["data"][ix]=np.amax(col,axis=1)
             """num_rows=col.shape[0]
             num_col=col.shape[1]
             index=0
             for j in range(num_col):
                 maximum=0
                 for i in range(num_rows):
                    if i==0:
                        maximum=col[i][j]
                    if col[i][j]>maximum:
                        maximum=col[i][j]
                 outputs["data"][ix][index]=maximum
                 index+=1  """                                       
        return outputs

    def backward(self, output_grads):
        """The backward pass

        Arguments:
            output_grads (OrderedDict): Containing
                grad: gradient wrt output
        Returns:
            input_grads (OrderedDict): Containing
                grad: gradient wrt input

        Note that we compute the output heights, widths, and
        channels on the fly in the backward pass as well.

        You may want to take a look at the im2col_conv and col2im_conv
        functions present in the base class ``BaseConvLayer``

        """
        input_grads = OrderedDict()
        input_grads["grad"] = np.zeros_like(self.data, dtype=dtype)
        h_in, w_in, c = self.h_in, self.w_in, self.c
        batch_size = self.data.shape[0]
        output_diff = output_grads["grad"]

        k = self.kernel_size

        h_out, w_out, c = self.get_output_dim(
            h_in, w_in, self.pad, self.stride,
            self.kernel_size, c
        )
    
        for ix in range(batch_size):
             col1 = self.im2col_conv(self.data[ix], h_in, w_in, c, h_out, w_out)
             col2 = self.im2col_conv(input_grads["grad"][ix], h_in, w_in, c, h_out, w_out)

             #reshaping col1 and col2
             #col1= col1.reshape(k*k,h_out*w_out*c)

             col1=col1.reshape(k*k,c,h_out*w_out)
             col1=col1.transpose(0,2,1)
             col1=col1.reshape(k*k,h_out*w_out*c)
             
             #col2= col2.reshape(k*k,h_out*w_out*c)

             col2=col2.reshape(k*k,c,h_out*w_out)
             col2=col2.transpose(0,2,1)
             col2=col2.reshape(k*k,h_out*w_out*c)
             
             output_ix= output_grads["grad"][ix]   
             for j in range(col1.shape[1]):
                 maximum=0
                 max_idx=-1
                 for i in range(col1.shape[0]):
                    if i==0:
                        maximum=col1[i][j]
                        max_idx=0
                    if col1[i][j]>maximum:
                        maximum=col1[i][j]
                        max_idx=i
                        
                 col2[max_idx][j]=output_ix[j]

             col2=col2.reshape(k*k,h_out*w_out,c)
             col2=col2.transpose(0,2,1)
             col2=col2.reshape(k*k*c,h_out*w_out)
             col2_im=self.col2im_conv(col2, h_in, w_in, c, h_out, w_out)
             '''print(col2_im.shape,"after resizing shape")
             print(input_grads["grad"][ix].shape,"orignal shape")
             print(col2.shape,"shape of col2")
             print(input_grads["grad"][ix].shape,"shape of the current input")
             print(col2_im.flatten().shape,"shape of col2 civerted back into input")'''
             input_grads["grad"][ix]=col2_im.flatten()
             
        return input_grads
