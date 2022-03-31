# ==================== Activations ====================
RELU    = 'relu'
SIGMOID = 'sigmoid'
SOFTMAX = 'softmax'
TANH    = 'tanh'
ACTIVATION_METHODS = [RELU, SIGMOID, SOFTMAX, TANH]


# ==================== Fusion Methods ====================
mlp_attribute_of_variable_length = ['dropouts', 'use_biases', 'activation_methods', 'use_batch_norms']


# ==================== Fusion Methods ====================
MEAN = 'mean'
FUSION_METHODS = [MEAN]


# ==================== Schedules ====================
BATCH_ALIGNMENT = 'batch_alignment'
CLASSIFICATION  = 'classification'
CLUSTERING      = 'clustering' 
TRANSLATION     = 'translation'
SCHEDULES = [BATCH_ALIGNMENT, CLASSIFICATION, CLUSTERING, TRANSLATION]


# ==================== Tasks ====================
CROSS_MODEL_PREDICTION = 'cross-model prediction'
SUPERVISED_GROUP_IDENTIFICATION = 'supervised group identification'
UNSUPERVISED_GROUP_IDENTIFICATION = 'unsupervised group identification'
TASKS = [CROSS_MODEL_PREDICTION, SUPERVISED_GROUP_IDENTIFICATION, UNSUPERVISED_GROUP_IDENTIFICATION]


# ==================== Groups ====================
EVALUATION = 'evaluation'
INFERENCE = 'inference'
TRAIN = 'train'


# ==================== Keys ====================
BATCH = 'batch'
LABEL = 'label'


# ==================== Attributes ====================
MODALITIES = 'modalities'
MODEL = 'model'