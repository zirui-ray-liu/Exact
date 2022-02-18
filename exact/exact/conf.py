class QuantizationConfig:
    def __init__(self):
        self.compress_activation = True
        self.activation_compression_bits = [2]
        self.initial_bits = 8
        self.stochastic = True
        self.use_gradient = False
        self.simulate = False
        self.enable_quantized_bn = True
        self.kept_frac = 1.0
        self.single_precision = True
        self.dropout2 = False
        # Memory management flag
        self.empty_cache_threshold = None
        self.swap = False
        self.amp = False
        # Debug related flag
        self.debug_memory_op_forward = False
        self.debug_memory_op_backward = False
        self.debug_remove_bn = False
        self.debug_remove_relu = False
        self.debug_remove_dropout = False

config = QuantizationConfig()