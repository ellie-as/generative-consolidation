from utils import prepare_data, noise, display
from initial_model import create_autoencoder
from initial_tests import check_initial_recall, iterative_recall
from generative_model import build_encoder_decoder, VAE
from generative_tests import interpolate_ims, plot_latent_space, check_generative_recall, plot_history
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import hopfield_utils
from hopfield_models import ContinuousHopfield
import matplotlib.backends.backend_pdf


def run_end_to_end(initial='autoencoder', generative='vae', dataset='mnist', initial_epochs=10, generative_epochs=10, num=100, latent_dim=5, kl_weighting=1, hopfield_type='continuous'):
    
    pdf = matplotlib.backends.backend_pdf.PdfPages("./outputs/output_{}items_{}_{}eps_{}_{}eps_{}lv.pdf".format(num, initial, initial_epochs, generative, generative_epochs, latent_dim))
    
    if dataset == 'mnist':
        train_data, test_data, noisy_train_data, noisy_test_data = prepare_data()
    else:
        print("Dataset not supported.")
        
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
        net = hopfield_utils.create_hopfield(num, hopfield_type=hopfield_type)
        predictions = []
        tests = []
        
        images = hopfield_utils.load_images_mnist(num)
        images_np = hopfield_utils.convert_images(images, 'black-white')
        images_masked_np = hopfield_utils.mask_image_random(images_np)
        images_np = [im_np.reshape(-1, 1) for im_np in images_np]
        images_masked_np = [im_np.reshape(-1, 1) for im_np in images_masked_np]
        
        for test_ind in range(num):
            test = images_masked_np[test_ind].reshape(-1,1)
            if hopfield_type is 'classical':
                reconstructed = net.retrieve(test)
            else:
                reconstructed = net.retrieve(test, max_iter=10)
            predictions.append(np.array(reconstructed).reshape(1,28,28,1))
            tests.append(np.array(test).reshape(1,28,28,1))
            
        print("Recalling noisy images with the initial model:")
        predictions = np.concatenate(predictions, axis=0)
        tests = np.concatenate(tests, axis=0)
        
        fig = display(tests,predictions, title='Inputs and outputs for Hopfield network')
        pdf.savefig(fig)
        predictions = np.where(predictions<0, 0, predictions)

    else:
        print("Initial model not supported.")
        
    if generative == 'vae':
        encoder, decoder = build_encoder_decoder(latent_dim = latent_dim)
        vae = VAE(encoder, decoder, kl_weighting)
        opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        vae.compile(optimizer=opt)
        history = vae.fit(predictions, epochs=generative_epochs, verbose=0)
    
        fig = plot_history(history)
        pdf.savefig(fig)
        
        print("Recalling noisy images with the generative model:")
        fig = check_generative_recall(vae, test_data)
        pdf.savefig(fig)
        
        print("Interpolating between image pairs:")
        latents = vae.encoder.predict(test_data)
        fig = interpolate_ims(latents, vae, 3, 6)
        pdf.savefig(fig)
        fig = interpolate_ims(latents, vae, 4, 8)
        pdf.savefig(fig)
        fig = interpolate_ims(latents, vae, 21, 32)
        pdf.savefig(fig)
        
        fig = plot_latent_space(vae, noisy_test_data)
        pdf.savefig(fig)
    else:
        print("Generative model not supported.")
        
    pdf.close()
        
    