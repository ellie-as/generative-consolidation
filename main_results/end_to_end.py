from utils import prepare_data, noise, display
from initial_model import create_autoencoder
from initial_tests import check_initial_recall, iterative_recall
from generative_model import VAE, build_encoder_decoder_v3, build_encoder_decoder_v5
from generative_tests import interpolate_ims, plot_latent_space, check_generative_recall, plot_history, vector_arithmetic
from tensorflow import keras
import numpy as np
from random import randrange
from PIL import Image
import matplotlib.pyplot as plt
import hopfield_utils
from hopfield_models import ContinuousHopfield
import matplotlib.backends.backend_pdf
from config import models_dict, dims_dict


def run_end_to_end(initial='autoencoder', generative='vae', dataset='mnist', initial_epochs=10, generative_epochs=10, num=100, latent_dim=5, kl_weighting=1, hopfield_type='continuous', lr=0.001, vector_arithmetic=True):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages("./outputs/output_{}_{}items_{}_{}eps_{}_{}eps_{}lv_{}lr_{}kl.pdf".format(dataset, num, initial, initial_epochs, generative, generative_epochs, latent_dim, lr, kl_weighting))
    
    train_data, test_data, noisy_train_data, noisy_test_data = prepare_data(dataset)
    
    dims = dims_dict[dataset]
    model = models_dict[dataset]
        
    if initial == 'autoencoder':
        autoencoder = create_autoencoder()

        autoencoder.fit(
            x=noisy_train_data,
            y=train_data,
            epochs=initial_epochs,
            batch_size=128,
            shuffle=True,
            validation_data=(noisy_test_data, test_data),
        )
        
        print("Recalling noisy images with the initial model:")
        predictions, fig = check_initial_recall(autoencoder, train_data)
        pdf.savefig(fig)
    
    elif initial == 'hopfield':
        net = hopfield_utils.create_hopfield(num, hopfield_type=hopfield_type, dataset=dataset)
        predictions = []
        tests = []
        
        images = hopfield_utils.load_images_mnist(num, dataset=dataset)

        images_np = hopfield_utils.convert_images(images, 'gray-scale')
        images_masked_np = hopfield_utils.mask_image_random(images_np)
        images_np = [im_np.reshape(-1, 1) for im_np in images_np]
        images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]
        
        for test_ind in range(num):
            test = images_masked_np[test_ind].reshape(-1,1)
            if hopfield_type is 'classical':
                reconstructed = net.retrieve(test)
            else:
                reconstructed = net.retrieve(test, max_iter=10)
            # if your image is greyscale, you need to pass PIL 2d array
            reshaped = np.array(reconstructed).reshape(1,dims[0],dims[1])
            test = np.array(test).reshape(1,dims[0],dims[1])
            predictions.append(reshaped.reshape(1,dims[0],dims[1],dims[2]))
            tests.append(test.reshape(1,dims[0],dims[1],dims[2]))
            
        print("Recalling noisy images with the initial model:")
        predictions = np.concatenate(predictions, axis=0)
        tests = np.concatenate(tests, axis=0)
        
        fig = display(tests,predictions, dataset=dataset, title='Inputs and outputs for Hopfield network')
        pdf.savefig(fig)
        # rescale predictions back to interval [0, 1]
        predictions = (predictions + 1) / 2 

    else:
        print("Initial model not supported.")
        
    if generative == 'vae':    
        
        encoder, decoder = models_dict[dataset](latent_dim = latent_dim)
        vae = VAE(encoder, decoder, kl_weighting)
        opt = keras.optimizers.Adam(lr=lr)
        vae.compile(optimizer=opt)
        history = vae.fit(predictions, epochs=generative_epochs, verbose=1, batch_size=32, shuffle=True)
    
        fig = plot_history(history)
        pdf.savefig(fig)
        
        print("Recalling noisy images with the generative model:")
        fig = check_generative_recall(vae, train_data[0:100])
        pdf.savefig(fig)
        
        print("Interpolating between image pairs:")
        latents = vae.encoder.predict(test_data)
        for i in range(10):
            fig = interpolate_ims(latents, vae, randrange(50), randrange(50))
            pdf.savefig(fig)
            
        if vector_arithmetic == True:
            for i in range(10):
                fig = vector_arithmetic(latents, vae, randrange(50), randrange(50), randrange(50))
                pdf.savefig(fig)
        
    else:
        print("Generative model not supported.")
        
    pdf.close()
        
    