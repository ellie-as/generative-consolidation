from generative_model import *

dims_dict = {"mnist": (28,28,1),
            "fashion_mnist": (128,128,1),
            "solids": (128,128,1),
            "shapes3d": (128,128,1),
            "kmnist": (28,28,1),
            "plant_village": (28,28,1)}

models_dict = {"mnist": build_encoder_decoder_v5,
            "fashion_mnist": build_encoder_decoder_v3,
            "solids": build_encoder_decoder_v3,
            "shapes3d": build_encoder_decoder_v3,
            "kmnist": build_encoder_decoder_v5,
            "plant_village": build_encoder_decoder_v5,}