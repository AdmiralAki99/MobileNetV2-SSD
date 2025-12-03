import tensorflow as tf

class LocalizationHead(tf.keras.Layer):
    def __init__(self,name: str, num_anchors_per_location: list[int], **kwargs):
        super().__init__(name=name)
        
        self.name = name
        
        self.heads = []
        self.head_type = kwargs['head_type']
        self.heads = []
        self.num_anchors_per_layer = num_anchors_per_location
        
        self.initial_norm = self.make_normalization(kwargs.get('initial_norm_strategy', "BatchNorm"))
        
        self.squeeze_heads = []
        self.squeeze_ratio = kwargs.get('squeeze_ratio',1.0)

        self.intermediate_channels = kwargs.get('intermediate_conv',None)
        self.intermediate_heads = [] if self.intermediate_channels is not None else None

    def call(self,feature_maps,training = False):
        
        if len(feature_maps) != len(self.num_anchors_per_layer):
            raise ValueError(
                f"{self.name}: got {len(feature_maps)} feature maps but "
                f"{len(self.num_anchors_per_layer)} anchor specs."
            )
            
        outputs = []
        for layer, feature_map in enumerate(feature_maps):
            num_anchors = self.num_anchors_per_layer[layer]
            
            # Getting the feature map
            x = feature_map

            # Initial Norm
            if self.initial_norm is not None and layer == 0:
                x = self.initial_norm(x,training = training)

            # Squeeze Layer
            if self.squeeze_ratio != 1.0:
                x = self.squeeze_heads[layer](x,training = training)
                
            # Intermediate Conv
            if self.intermediate_heads is not None:
                x = self.intermediate_heads[layer](x, training=training)
            
            # Prediction Conv
            x = self.heads[layer](x,training = training)

            # Reshape
            B = tf.shape(x)[0]
            H = tf.shape(x)[1]
            W = tf.shape(x)[2]

            x = tf.reshape(x, [B, H, W, num_anchors, 4])
            x = tf.reshape(x, [B, H * W * num_anchors, 4])

            # Append the value
            outputs.append(x)

        # Concatenate
        final_output = tf.concat(outputs,axis=1)
        return final_output

    def build(self,input_shape):
        for layer,feature_map_shape in enumerate(input_shape):
            channel = int(feature_map_shape[-1])

            # Need to calculate the squeeze heads
            if self.squeeze_ratio != 1.0:
                squeeze_out = int(channel * self.squeeze_ratio)
                squeeze = self.make_squeeze_head(squeeze_out, index=layer)
                self.squeeze_heads.append(squeeze)
                input_channels_for_pred = squeeze_out
            else:
                input_channels_for_pred = channel

            # Need to calculate the intermediate heads
            if self.intermediate_channels is not None:
                intermediate_head = self.make_head(self.head_type, self.intermediate_channels, layer,role="inter")
                self.intermediate_heads.append(intermediate_head)

            A_per_layer = self.num_anchors_per_layer[layer]
            
            pred_head = self.make_head(self.head_type, A_per_layer * 4, index = layer, role = "pred")
            self.heads.append(pred_head)
        
    def make_head(self,head_type: str, out_channels: int, index: int, role: str):
        base = f"{self.name}_loc_{role}_{index}"
        if head_type == "conv3x3":
            return tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3,padding="same",name=base)
        elif head_type == "depthwise":
            dw_name = f"{base}_dw"
            pw_name = f"{base}_pw"
            return tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(kernel_size = 3, padding="same",name=dw_name),
                tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,padding="same",name=pw_name)
            ],name=base)

    def make_squeeze_head(self,out_channels: int,index: int):
        base = f"{self.name}_loc_squeeze_{index}"
        return tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,padding="same",name=base)

    def make_normalization(self, normalization_type):
        if normalization_type == "BatchNorm":
            return tf.keras.layers.BatchNormalization(name = "loc_initial_normalization")
        elif normalization_type == "Norm":
            return tf.keras.layers.Normalization(name = "initial_normalization")
        
     
class ClassificationHead(tf.keras.Layer):
    def __init__(self,name: str, num_anchors_per_location: list[int], number_of_classes: int, norm_cfg: str = "BatchNorm", head_type: str = "conv3x3",use_sigmoid: bool = False, **kwargs):
        super().__init__(name=name)
        
        self.name = name

        # Stored the number of the classes
        self.number_of_classes = number_of_classes

        # Stored the head type
        self.head_type = head_type

        # Stored the anchors per layer
        self.num_anchors_per_location = num_anchors_per_location

        # Initial normalization strategy
        self.initial_norm = self.make_normalization(norm_cfg)

        # Squeeze Ratio
        self.squeeze_ratio = kwargs.get('squeeze_ratio',1.0)
        self.squeeze_blocks = []            

        # Intermediate Conv blocks
        self.intermediate_channels = kwargs.get('intermediate_conv',None)
        self.intermediate_blocks = [] if self.intermediate_channels is not None else None
        
        # Creating the final pred values
        self.final_heads = []
        
        self.use_sigmoid = use_sigmoid

    def make_normalization(self, normalization_type):
        if normalization_type == "BatchNorm":
            return tf.keras.layers.BatchNormalization(name = "cls_initial_normalization")
        elif normalization_type == "Norm":
            return tf.keras.layers.Normalization(name = "cls_initial_normalization")

    def call(self,feature_maps, training = False):
        if len(feature_maps) != len(self.num_anchors_per_location):
            raise ValueError(
                f"{self.name}: got {len(feature_maps)} feature maps but "
                f"{len(self.num_anchors_per_location)} anchor specs."
            )
        outputs = []
        for layer, feature_map in enumerate(feature_maps):
            num_anchors = self.num_anchors_per_location[layer]
            C = self.number_of_classes
            x = feature_map

            # Initial Normalization Strategy
            if self.initial_norm != None and layer == 0:
                x = self.initial_norm(x,training = training)

            # Squeeze Ratio
            if self.squeeze_ratio != 1.0:
                x = self.squeeze_blocks[layer](x,training=training)

            # Intermediate Conv
            if self.intermediate_blocks != None:
                 x = self.intermediate_blocks[layer](x,training = training)

            # Final Predection
            x = self.final_heads[layer](x,training=training)

            # Reshape
            B = tf.shape(x)[0]
            H = tf.shape(x)[1]
            W = tf.shape(x)[2]

            # The shape must be (B,H,W,A*C)
            x = tf.reshape(x,[B,H,W,num_anchors,C])
            x = tf.reshape(x,[B,H * W * num_anchors,C])
            outputs.append(x)

        return tf.concat(outputs,axis=1)

    def build(self,input_shape):
        for layer,feature_map_shape in enumerate(input_shape):
            channel = int(feature_map_shape[-1])
            # Calculating the squeeze heads
            if self.squeeze_ratio != 1.0:
                squeeze_out = int(channel * self.squeeze_ratio)
                squeeze = self.create_head(out_channels = squeeze_out, index = layer, role="squeeze")
                self.squeeze_blocks.append(squeeze)
                input_channels_for_pred = squeeze_out
            else:
                input_channels_for_pred = channel

            # Calculate the intermediate heads
            if self.intermediate_channels is not None: 
                intermediate_head = self.create_head(out_channels = self.intermediate_channels, index = layer, role="inter")
                self.intermediate_blocks.append(intermediate_head)

            A_per_layer = self.num_anchors_per_location[layer]

            # Create final head
            pred_head = self.create_head(out_channels = A_per_layer * self.number_of_classes, index = layer, role = "pred")
            self.final_heads.append(pred_head)

    def create_pred_heads(self,anchors_per_layer: list[int]):
        heads = []
        for layer_number, anchors in enumerate(anchors_per_layer):
            # Creating the head based on the formula of the Ai * C
            head = self.create_head(anchors * self.number_of_classes,layer_number,"pred")
            heads.append(head)

        return heads

    def create_intermediate_heads(self,anchors_per_layer: list[int]):
        heads = []
        for layer_number in range(len(anchors_per_layer)):
            # Creating the head based on the formula of the Ai * C
            head = self.create_head(self.intermediate_channels,layer_number,"intermediate")
            heads.append(head)

        return heads

    def create_squeeze_heads(self,channels_per_layer: list[int]):
        heads = []
        for layer_number, channels in enumerate(channels_per_layer):
            # Creating the head based on the formula of the Ai * C
            out_channel = int(channels * self.squeeze_ratio)
            head = self.create_head(out_channel,layer_number,"squeeze")
            heads.append(head)

        return heads

    def create_head(self, out_channels: int, index: int, role: str):
        base = f"{self.name}_cls_{role}_{index}"
        if self.head_type == "conv3x3":
            return tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3,padding="same",name=base)
        elif self.head_type == "depthwise":
            dw_name = f"{base}_dw"
            pw_name = f"{base}_pw"
            return tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(kernel_size = 3, padding="same",name=dw_name),
                tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,padding="same",name=pw_name)
            ],name=base)
        