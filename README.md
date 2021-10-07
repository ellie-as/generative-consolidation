
### Consolidation simulations

Work-in-progress code for modelling consolidation as teacher-student learning, in which initial representations of memories are replayed to train a generative model.

To use this code, install the requirements and launch a jupyter notebook (or alternatively use AWS SageMaker, e.g. the conda_amazonei_tensorflow2_p36 kernel).

#### End-to-end simulation example

The following code snippet:
* Trains a modern Hopfield network on the MNIST dataset of handwritten digits
* Gives the Hopfield network random noise as an input, and gets the outputs (which should be memories)
* Trains a variational autoencoder on the 'memories'
* Runs a set of tests, e.g. tests recall and interpolation between items, and plots the latent space projected into 2D
* Saves the outputs to a pdf in the 'outputs' folder (see example)


```python
from end_to_end import run_end_to_end
import tensorflow as tf

# set tensorflow random seed to make outputs reproducible
tf.random.set_seed(123)
```

The cell below recreates the results in the 'outputs' folder:


```python
run_end_to_end(initial='hopfield', generative='vae', dataset='shapes3d', generative_epochs=1000, 
               num=1000, latent_dim=10, kl_weighting=1)

run_end_to_end(initial='hopfield', generative='vae', dataset='solids', generative_epochs=1000, 
               num=1000, latent_dim=10, kl_weighting=1)

run_end_to_end(initial='hopfield', generative='vae', dataset='fashion_mnist', generative_epochs=1000, 
               num=1000, latent_dim=10, kl_weighting=1)

run_end_to_end(initial='hopfield', generative='vae', dataset='mnist', generative_epochs=1000, 
               num=1000, latent_dim=10, kl_weighting=1)
```

The options can be swapped out to run different experiments, e.g. to try with an autoencoder as the initial model:


```python
run_end_to_end(initial='autoencoder', generative='vae', dataset='mnist', initial_epochs=10, generative_epochs=10)
```

#### Other examples

Prepare the training data:


```python
train_data, test_data, noisy_train_data, noisy_test_data = prepare_data()
```

Create a modern Hopfield network and store 1000 MNIST memories:


```python
net = hopfield_utils.create_hopfield(1000, hopfield_type='continuous')
```

Alternatively, build and fit a denoising autoencoder:


```python
autoencoder = create_autoencoder()

autoencoder.fit(
    x=noisy_train_data,
    y=train_data,
    epochs=initial_epochs,
    batch_size=128,
    shuffle=True,
    validation_data=(noisy_test_data, test_data),
)
```

Display recall from random noise by the initial model:


```python
predictions, fig = check_initial_recall(autoencoder, train_data)
```

Train a variational autoencoder on replayed memories from the initial model (i.e. outputs when the initial model is presented with random noise), and plot the loss over time:


```python
encoder, decoder = build_encoder_decoder(latent_dim = 5)
vae = VAE(encoder, decoder, kl_weighting=1)
opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
vae.compile(optimizer=opt)
history = vae.fit(predictions, epochs=generative_epochs, verbose=0)

fig = plot_history(history)
```

The Hopfield network code is based on https://github.com/ml-jku/hopfield-layers
The variational autoencoder code is based on https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py
