import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, Add, ZeroPadding2D,AveragePooling2D, ReLU, DepthwiseConv2D, GlobalAveragePooling2D

class StandardConvolutionBlock(tf.keras.Model):
    def __init__(self,output_channels = 32,stride = 2,alpha = 1.0,name = None):
        super().__init__(name=name)
        output_channel = max(1,int(output_channels*alpha)) # To scale the proportions to make them fit for different devices
        self.conv_layer = Conv2D(filters = output_channel, kernel_size = 3, strides = stride, padding='same', kernel_initializer = "he_normal", use_bias = False, name = None if name is None else f"{name}_conv")
        self.batch_norm_layer = BatchNormalization(name = f"{name}_bn" if name else None)
        self.activation_layer = ReLU(max_value = 6.0,name = f"{name}_relu6" if name else None)
        
    def call(self,x,training = False):
        # Standard block takes a convolution, then batch normalizes it and then puts it through a ReLU6
        x = self.conv_layer(x,training = training)
        x = self.batch_norm_layer(x, training = training)
        x = self.activation_layer(x, training = training)

        return x

    def _find_ref(self, reference_table, name_dict, lookup_name):
        # First check if it is a valid value
        if lookup_name not in name_dict:
            raise KeyError( f"Reference Name not found: need '{name_dict}' "f"(mapped from '{lookup_name}')")
            
        # Finding the layer to lookup
        lookup_layer = reference_table.get(name_dict[lookup_name],lookup_name)
        if type(lookup_layer) == str:
            raise KeyError( f"Reference layer not found: need '{reference_table}' "f"(mapped from '{lookup_name}')")

        return lookup_layer

    def transplant_weights(self, reference_table : dict, name_dict : dict = None):
        self._assign_conv_weights(self.conv_layer,self._find_ref(reference_table,name_dict,'Conv1_conv'))
        self._assign_bn_weights(self.batch_norm_layer,self._find_ref(reference_table,name_dict,'Conv1_bn'))

    def _assign_conv_weights(self, destination, source):
        destination_weights = destination.weights
        source_weights = source.weights

        # Checking if the lengths are the same
        assert len(destination_weights) == len(source_weights), ( f'{destination.name}: variable count mismatch {len(destination_weights)} != {len(source_weights)}')

        for dv,sv in zip(destination_weights,source_weights):
            # Checking if the shape is the same
            assert dv.shape == sv.shape, (f"{destination.name}: shape mismatch {dv.shape} != {sv.shape}")
            dv.assign(sv)

    def _assign_bn_weights(self,destination, source):
        destination_weights = destination.weights
        source_weights = source.weights

        assert len(destination_weights) == len(source_weights), ( f'{destination.name}: variable count mismatch {len(destination_weights)} != {len(source_weights)}')

        for dv,sv in zip(destination_weights,source_weights):
            # Checking if the shape is the same
            assert dv.shape == sv.shape, (f"{destination.name}: shape mismatch {dv.shape} != {sv.shape}")
            dv.assign(sv)
            
            

class InvertedResidualBlock(tf.keras.Model):
    def __init__(self,output_channels = 32, expansion_factor = 6, stride = 2, alpha = 1.0,name = None):
        super().__init__(name= name)
        self.output_channel = output_channels
        self.expansion_channel = None
        self.expansion_factor = expansion_factor
        self.name = name
        self.stride = stride
        self.alpha = alpha
        # Creating the layers inside that are dependent on the input shape
        self.expansion_conv = None
        self.expand_batch_norm = None
        self.expand_activation_function = None

        self.depthwise_conv = None
        self.depthwise_batch_norm = None
        self.depthwise_activation_function = None

        self.projection_conv = None
        self.project_batch_norm = None

    def build(self,input_shape):
        # Building the block here
        input_channel = int(input_shape[-1]) 
        self.output_channel =  self._make_divisible(int(round(self.output_channel * self.alpha)), 8)
        self.expansion_channel = int(input_channel * self.expansion_factor)

        # Now in the sequence there is one case where the bottleneck does not expand
        if self.expansion_factor != 1:
            # Expansion layer needs to be created
            self.expansion_conv = Conv2D(self.expansion_channel, kernel_size = 1, strides = 1, padding="same", use_bias = False, kernel_initializer = "he_normal", name = None if self.name is None else f"{self.name}_expand_conv") 
            self.expand_batch_norm = BatchNormalization(name = None if self.name is None else f"{self.name}_expand_bn")
            self.expand_activation_function = ReLU(max_value = 6.0, name = None if self.name is None else f"{self.name}_expand_relu6")

        else:
            # There is not expansion to be carried out
            self.expansion_channel = input_channel

        # Depthwise Conv Layer

        self.depthwise_conv = DepthwiseConv2D(kernel_size = 3, strides = self.stride, padding = "same", use_bias = False, depthwise_initializer="he_normal", name = None if self.name is None else f"{self.name}_dw_conv")
        self.depthwise_batch_norm = BatchNormalization(name = None if self.name is None else f"{self.name}_dw_bn")
        self.depthwise_activation_function = ReLU(max_value = 6.0, name = None if self.name is None else f"{self.name}_dw_relu6")

        self.projection_conv = Conv2D(self.output_channel, kernel_size = 1, strides = 1, padding="same", use_bias = False, kernel_initializer = "he_normal", name = None if self.name is None else f"{self.name}_project_conv")
        self.project_batch_norm = BatchNormalization(name = None if self.name is None else f"{self.name}_project_bn")

        super().build(input_shape)

    def call(self,x,training = False):
        if self.expansion_conv is not None:
            x = self.expansion_conv(x,training = training)
            x = self.expand_batch_norm(x, training = training)
            x = self.expand_activation_function(x)

        x = self.depthwise_conv(x)
        x = self.depthwise_batch_norm(x, training = training)
        x = self.depthwise_activation_function(x)

        x = self.projection_conv(x,training = training)
        x = self.project_batch_norm(x,training = training)

        return x

    def transplant_weights(self, reference_table : dict, name_dict : dict = None):
        ## There are two cases to handle, if there is an expansion factor or not
        if self.expansion_factor == 1:
            self._assign_conv_weights(self.depthwise_conv,self._find_ref(reference_table,name_dict,f"{self.name}_dw_conv"))
            self._assign_bn_weights(self.depthwise_batch_norm,self._find_ref(reference_table,name_dict,f"{self.name}_dw_bn"))
            self._assign_conv_weights(self.projection_conv,self._find_ref(reference_table,name_dict,f"{self.name}_project_conv"))
            self._assign_bn_weights(self.project_batch_norm,self._find_ref(reference_table,name_dict,f"{self.name}_project_bn"))
        else:
            self._assign_conv_weights(self.expansion_conv,self._find_ref(reference_table,name_dict,f"{self.name}_expand_conv"))
            self._assign_bn_weights(self.expand_batch_norm,self._find_ref(reference_table,name_dict, f"{self.name}_expand_bn"))
            self._assign_conv_weights(self.depthwise_conv,self._find_ref(reference_table,name_dict,f"{self.name}_dw_conv"))
            self._assign_bn_weights(self.depthwise_batch_norm,self._find_ref(reference_table,name_dict,f"{self.name}_dw_bn"))
            self._assign_conv_weights(self.projection_conv,self._find_ref(reference_table,name_dict,f"{self.name}_project_conv"))
            self._assign_bn_weights(self.project_batch_norm,self._find_ref(reference_table,name_dict,f"{self.name}_project_bn"))

        

    def _assign_conv_weights(self, destination, source):
        destination_weights = destination.weights
        source_weights = source.weights

        # Checking if the lengths are the same
        assert len(destination_weights) == len(source_weights), ( f'{destination.name}: variable count mismatch {len(destination_weights)} != {len(source_weights)}')

        for dv,sv in zip(destination_weights,source_weights):
            # Checking if the shape is the same
            assert dv.shape == sv.shape, (f"{destination.name}: shape mismatch {dv.shape} != {sv.shape}")
            dv.assign(sv)

        # print(f"{destination.name}: Weight Transplant Success!")

    def _assign_bn_weights(self,destination, source):
        destination_weights = destination.weights
        source_weights = source.weights

        assert len(destination_weights) == len(source_weights), ( f'{destination.name}: variable count mismatch {len(destination_weights)} != {len(source_weights)}')

        for dv,sv in zip(destination_weights,source_weights):
            # Checking if the shape is the same
            assert dv.shape == sv.shape, (f"{destination.name}: shape mismatch {dv.shape} != {sv.shape}")
            dv.assign(sv)

        # print(f"{destination.name}: Weight Transplant Success!")

    def _find_ref(self, reference_table, name_dict, lookup_name):
        # First check if it is a valid value
        if lookup_name not in name_dict:
            raise KeyError( f"Reference Name not found: need '{name_dict}' "f"(mapped from '{lookup_name}')")
            
        # Finding the layer to lookup
        lookup_layer = reference_table.get(name_dict[lookup_name],lookup_name)
        if type(lookup_layer) == str:
            raise KeyError( f"Reference layer not found: need '{reference_table}' "f"(mapped from '{lookup_name}')")

        return lookup_layer
    
    def _make_divisible(self,v, divisor=8):
        return max(divisor, int(v + divisor / 2) // divisor * divisor)