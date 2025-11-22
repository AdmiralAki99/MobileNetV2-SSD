import tensorflow as tf

class LocalizationHead(tf.keras.Model):
    def __init__(self,name: str, num_anchors_per_location: list[int], **kwargs):
        super().__init__(name=name)
        
        self.heads = []
        self.head_type = kwargs['head_type']
        self.heads = self.make_heads(num_anchors_per_location)
        self.num_anchors_per_layer = num_anchors_per_location
        
        if 'initial_norm_strategy' in kwargs:
            self.initial_norm = self.make_normalization(kwargs['initial_norm_strategy'])
        else:
            self.initial_norm = None
        
        self.squeeze_heads = None
        if 'squeeze_ratio' in kwargs:
            self.squeeze_ratio = kwargs['squeeze_ratio']
            self.squeeze_heads = self.make_squeeze_heads(kwargs['in_channels'])
        else:
            self.squeeze_ratio = 1.0
            
        self.intermediate_heads = None
        if 'intermediate_conv' in kwargs:
            self.intermediate_channels = kwargs['intermediate_conv']
            self.intermediate_heads = self.make_intermediate_heads(num_anchors_per_location)

    def call(self,feature_maps,training = False):
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

    def make_heads(self,anchors_per_location: list[int]):
        heads = []
        for layer, anchors in enumerate(anchors_per_location):
            output_channel = anchors * 4
            head = self.make_head(self.head_type,output_channel,layer,role="pred")
            heads.append(head)

        return heads

    def make_squeeze_head(self,out_channels: int,index: int):
        base = f"{self.name}_loc_squeeze_{index}"
        return tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,padding="same",name=base)

    def make_intermediate_heads(self, anchors_per_location: list[int]):
        heads = []
        for layer in range(len(anchors_per_location)):
            heads.append(self.make_head(self.head_type, self.intermediate_channels,layer,role="inter"))
        
        return heads

    def make_squeeze_heads(self, channels_per_location):
        heads = []
        for layer, channels in enumerate(channels_per_location):
            output_channel = int(channels * self.squeeze_ratio)
            head = self.make_squeeze_head(output_channel,layer)
            heads.append(head)

        return heads

    def make_normalization(self, normalization_type):
        if normalization_type == "BatchNorm":
            return tf.keras.layers.BatchNormalization(name = "loc_initial_normalization")
        elif normalization_type == "Norm":
            return tf.keras.layers.Normalization(name = "initial_normalization")
        
        raise ValueError(f"Normalization type {normalization_type} not recognized.")

      
class ClassificationHead(tf.keras.Layer):
    def __init__(self,name: str, num_anchors_per_location: list[int], number_of_classes: int, norm_cfg: str, head_type: str = "conv3x3",use_sigmoid: bool = False, **kwargs):
        super().__init__(name=name)

        # Stored the number of the classes
        self.number_of_classes = number_of_classes

        # Stored the head type
        self.head_type = head_type

        # Stored the anchors per layer
        self.num_anchors_per_location = num_anchors_per_location

        # Initial normalization strategy
        self.initial_norm = None
        self.initial_norm = self.make_normalization(norm_cfg)

        # Squeeze Ratio
        self.squeeze_ratio = 1.0
        self.squeeze_blocks = None
        if 'squeeze_ratio' in kwargs:
            self.squeeze_ratio = kwargs['squeeze_ratio']
            self.squeeze_blocks = self.create_squeeze_heads(kwargs['in_channels'])
            

        # Intermediate Conv blocks
        self.intermediate_blocks = None
        self.intermediate_channels = None
        if 'intermediate_channels' in kwargs:
            self.intermediate_channels = kwargs['intermediate_channels']
            self.intermediate_blocks = self.create_intermediate_heads(self.num_anchors_per_location)
        
        # Creating the final pred values
        self.final_heads = []
        self.final_heads = self.create_pred_heads(self.num_anchors_per_location)
        pass

    def make_normalization(self, normalization_type):
        if normalization_type == "BatchNorm":
            return tf.keras.layers.BatchNormalization(name = "loc_initial_normalization")
        elif normalization_type == "Norm":
            return tf.keras.layers.Normalization(name = "initial_normalization")

    def call(self,feature_maps, training = False):
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
        elif self.head_type == "dw":
            dw_name = f"{base}_dw"
            pw_name = f"{base}_pw"
            return tf.keras.Sequential([
                tf.keras.layers.DepthwiseConv2D(kernel_size = 3, padding="same",name=dw_name),
                tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,padding="same",name=pw_name)
            ],name=base)
        