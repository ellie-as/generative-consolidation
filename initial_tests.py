from utils import noise, display

def check_initial_recall(autoencoder, train_data, noise_level=0.2):
    noisy_train_data = noise(train_data, noise_factor=noise_level)
    predictions = autoencoder.predict(noisy_train_data)
    fig = display(noisy_train_data, predictions)
    return predictions, fig
    
def iterative_recall(autoencoder, test_data, noise_factor, num_iter=3):
    noisy_test_data = noise(test_data, noise_factor=noise_factor)
    random_predictions = autoencoder.predict(noisy_test_data)
    display(noisy_test_data, random_predictions, seed=5)

    for i in range(num_iter):
        random_predictions_old = random_predictions
        random_predictions = autoencoder.predict(random_predictions)
        display(random_predictions_old, random_predictions, seed=5)
    
    return random_predictions