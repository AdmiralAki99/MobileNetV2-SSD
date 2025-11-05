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
    
    def _make_divisible(v, divisor=8):
        return max(divisor, int(v + divisor / 2) // divisor * divisor)

class MobileNetV2(tf.keras.Model):
    def __init__(self,number_of_classes,name="backbone", alpha = 1.0, **kwargs):
        super().__init__(name=name, **kwargs)

        # Creating the layers in the model
        # This is the standard convolutional block
        self.conv_1 = StandardConvolutionBlock(output_channels = 32,stride = 2,alpha = alpha,name = "Conv1")

        # This is the start of the bottleneck blocks
        self.bottleneck_1 = InvertedResidualBlock(output_channels=16, stride = 1, expansion_factor = 1, alpha = alpha, name="bottleneck_block_1")
        self.bottleneck_2 = InvertedResidualBlock(output_channels=24, stride = 2, expansion_factor = 6, alpha = alpha, name="bottleneck_block_2")
        self.bottleneck_3 = InvertedResidualBlock(output_channels=24, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_3")
        self.bottleneck_4 = InvertedResidualBlock(output_channels=32, stride = 2, expansion_factor = 6, alpha = alpha, name="bottleneck_block_4")
        self.bottleneck_5 = InvertedResidualBlock(output_channels=32, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_5")
        self.bottleneck_6 = InvertedResidualBlock(output_channels=32, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_6")
        self.bottleneck_7 = InvertedResidualBlock(output_channels=64, stride = 2, expansion_factor = 6, alpha = alpha, name="bottleneck_block_7")
        self.bottleneck_8 = InvertedResidualBlock(output_channels=64, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_8")
        self.bottleneck_9 = InvertedResidualBlock(output_channels=64, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_9")
        self.bottleneck_10 = InvertedResidualBlock(output_channels=64, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_10")
        self.bottleneck_11 = InvertedResidualBlock(output_channels=96, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_11")
        self.bottleneck_12 = InvertedResidualBlock(output_channels=96, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_12")
        self.bottleneck_13 = InvertedResidualBlock(output_channels=96, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_13")
        self.bottleneck_14 = InvertedResidualBlock(output_channels=160, stride = 2, expansion_factor = 6, alpha = alpha, name="bottleneck_block_14")
        self.bottleneck_15 = InvertedResidualBlock(output_channels=160, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_15")
        self.bottleneck_16 = InvertedResidualBlock(output_channels=160, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_16")
        self.bottleneck_17 = InvertedResidualBlock(output_channels=320, stride = 1, expansion_factor = 6, alpha = alpha, name="bottleneck_block_17")
        self.conv_2 = Conv2D(1280, kernel_size=1, strides=1, padding="same",use_bias=False, name="conv_head")
        self.final_batch_norm = BatchNormalization(name="conv_head_bn")
        self.final_relu = ReLU(6.0, name="conv_head_relu")

    def call(self,x, training = False, **kwargs):

        x = self.conv_1(x)

        # Bottleneck blocks
        x = self.bottleneck_1(x, training = training)
        x = self.bottleneck_2(x, training = training)
        skip = x
        x = self.bottleneck_3(x, training = training)
        x = Add()([skip,x])
        x = self.bottleneck_4(x, training = training)
        skip = x
        x = self.bottleneck_5(x, training = training)
        x = Add()([skip,x])
        skip = x
        x = self.bottleneck_6(x, training = training)
        x = Add()([skip,x])
        x = self.bottleneck_7(x, training = training)
        skip = x
        x = self.bottleneck_8(x, training = training)
        x = Add()([skip,x])
        skip = x
        x = self.bottleneck_9(x, training = training)
        x = Add()([skip,x])
        skip = x
        x = self.bottleneck_10(x, training = training)
        x = Add()([skip,x])
        x = self.bottleneck_11(x, training = training)
        skip = x
        x = self.bottleneck_12(x, training = training)
        x = Add()([skip,x])
        skip = x
        x = self.bottleneck_13(x, training = training)
        x = Add()([skip,x])
        x = self.bottleneck_14(x, training = training)
        skip = x
        x = self.bottleneck_15(x, training = training)
        x = Add()([skip,x])
        skip = x
        x = self.bottleneck_16(x, training = training)
        x = Add()([skip,x])
        x =  self.bottleneck_17(x, training = training)
        
        x = self.conv_2(x)
        x = self.final_batch_norm(x, training = training)
        x = self.final_relu(x, training = training)

        return x

    def transplant_weights(self, reference_table, model_maps):

        # Now to isolate the name_maps for each of the blocks and then passing it to them
        self.conv_1.transplant_weights(reference_table,model_maps['standard_conv_block'])
        self.bottleneck_1.transplant_weights(reference_table,model_maps['bottlenecks'][1])
        self.bottleneck_2.transplant_weights(reference_table,model_maps['bottlenecks'][2])
        self.bottleneck_3.transplant_weights(reference_table,model_maps['bottlenecks'][3])
        self.bottleneck_4.transplant_weights(reference_table,model_maps['bottlenecks'][4])
        self.bottleneck_5.transplant_weights(reference_table,model_maps['bottlenecks'][5])
        self.bottleneck_6.transplant_weights(reference_table,model_maps['bottlenecks'][6])
        self.bottleneck_7.transplant_weights(reference_table,model_maps['bottlenecks'][7])
        self.bottleneck_8.transplant_weights(reference_table,model_maps['bottlenecks'][8])
        self.bottleneck_9.transplant_weights(reference_table,model_maps['bottlenecks'][9])
        self.bottleneck_10.transplant_weights(reference_table,model_maps['bottlenecks'][10])
        self.bottleneck_11.transplant_weights(reference_table,model_maps['bottlenecks'][11])
        self.bottleneck_12.transplant_weights(reference_table,model_maps['bottlenecks'][12])
        self.bottleneck_13.transplant_weights(reference_table,model_maps['bottlenecks'][13])
        self.bottleneck_14.transplant_weights(reference_table,model_maps['bottlenecks'][14])
        self.bottleneck_15.transplant_weights(reference_table,model_maps['bottlenecks'][15])
        self.bottleneck_16.transplant_weights(reference_table,model_maps['bottlenecks'][16])
        self.bottleneck_17.transplant_weights(reference_table,model_maps['bottlenecks'][17])
        self._assign_final_layer_weights(reference_table,model_maps['final_block'])

    def _assign_final_layer_weights(self, reference_table : dict, name_dict : dict = None):
        self._assign_conv_weights(self.conv_2,self._find_ref(reference_table,name_dict,"conv_head"))
        self._assign_bn_weights(self.final_batch_norm,self._find_ref(reference_table,name_dict,"conv_head_bn"))
        

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

    def _apply_residual(self,x,y,name):
        # Checking if the shape of the two values is the same
        input_shape = x.shape
        output_shape = y.shape
        same_axis = (input_shape[1] == output_shape[1]) and (input_shape[2] == output_shape[2])
        same_channels = (input_shape[-1] == output_shape[-1])

        if same_axis and same_channels:
            # There is no issue and they can be added
            return Add(name=name)([x,y])
        else:
            raise ValueError(f"There is a mismatch between shapes: {input_shape} and {output_shape}")