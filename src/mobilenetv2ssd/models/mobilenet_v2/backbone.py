import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, ReLU

from blocks import StandardConvolutionBlock, InvertedResidualBlock

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
        
        
# Builder Function
def build_mobilenetv2_backbone(input_shape=(224,224,3), alpha=1.0, name="mobilenetv2_backbone"):
    input_layer = Input(shape=input_shape)
    mobilenetv2 = MobileNetV2(number_of_classes=None, alpha=alpha, name=name)
    mobilenetv2.call(input_layer)
    return mobilenetv2

# Function to load the weights from a reference model
def load_mobilenetv2_weights(model: MobileNetV2, weights_path: str):
    # Loading the model weights from the given path
    model = model.load_weights(weights_path)
    return model