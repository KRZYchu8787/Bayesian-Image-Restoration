import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_grayscale_image(path):
    """
    Load a grayscale image and convert it to a numpy array.
    :param path: str, path to the image file
    :return: numpy.ndarray, normalized grayscale image
    """
    image = Image.open(path).convert('L')
    return np.asarray(image, dtype=np.float32) / 255.0

# Add Gaussian noise
def add_noise(img, sigma=0.1):
    """
    Add Gaussian noise to an image.
    :param img: numpy.ndarray, input image
    :param sigma: float, standard deviation of the Gaussian noise
    :return: numpy.ndarray, noisy image
    """
    noisy_image = img + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy_image, 0, 1)

def neighbors(i, j, shape):
    """
    Generate the coordinates of the neighboring pixels for a given pixel (i, j).
    :param i: int, row index of the pixel
    :param j: int, column index of the pixel
    :param shape: tuple, shape of the image (height, width)
    :return: generator, yields (ni, nj) for neighboring pixels
    """
    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
            yield ni, nj

def quadratic_local_energy(x, i, j, val, lam, alpha):
    """
    Calculate the local energy for the quadratic energy model at pixel (i, j) with value `val`.
    :param x: numpy.ndarray, current state of the image
    :param i: int, row index of the pixel
    :param j: int, column index of the pixel
    :param val: float, value to evaluate (0 or 1)
    :param lam: float, regularization parameter
    :param alpha: float, threshold for energy calculation
    :return: float, local energy contribution
    """
    energy = 0
    for ni, nj in neighbors(i, j, x.shape):
        diff = lam * (val - x[ni, nj])
        energy += min(diff**2, alpha)
    return energy

def gibbs_sampler_quadratic(x, y, sigma, lam, alpha):
    """
    Perform one step of Gibbs sampling for the quadratic energy model.
    :param x: numpy.ndarray, current state of the image
    :param y: numpy.ndarray, observed noisy image
    :param sigma: float, standard deviation of the noise
    :param lam: float, regularization parameter
    :param alpha: float, threshold for energy calculation
    :return: numpy.ndarray, updated state of the image
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vals = [0.0, 1.0]
            probs = []
            for val in vals:
                data_term = (y[i, j] - val) ** 2 / (2 * sigma**2)
                prior_term = quadratic_local_energy(x, i, j, val, lam, alpha)
                energy = data_term + prior_term
                probs.append(np.exp(-energy))
            probs = np.array(probs)
            probs /= np.sum(probs)
            x[i, j] = np.random.choice(vals, p=probs)
    return x

def simulated_annealing_quadratic(y, n_iter=50, sigma=0.1, lam=2.0, alpha=1.0, cooling=0.95):
    """
    Perform simulated annealing for the quadratic energy model.
    :param y: numpy.ndarray, observed noisy image
    :param n_iter: int, number of iterations
    :param sigma: float, standard deviation of the noise
    :param lam: float, regularization parameter
    :param alpha: float, threshold for energy calculation
    :param cooling: float, cooling factor for annealing
    :return: numpy.ndarray, denoised image
    """
    x = np.round(y.copy())
    for t in range(n_iter):
        x = gibbs_sampler_quadratic(x, y, sigma, lam, alpha)
        lam *= cooling
    return x

def map_estimate(x):
    """
    Perform MAP estimation on the input image.
    :param x: numpy.ndarray, input image
    :return: numpy.ndarray, MAP estimate of the image
    """
    return x

def mms_estimate(samples):
    """
    Perform MMS estimation on a collection of samples.
    :param samples: list of numpy.ndarray, collection of samples
    :return: numpy.ndarray, MMS estimate of the image
    """
    return np.mean(samples, axis=0)

def potts_local_energy(x, i, j, val, beta):
    """
    Calculate the local energy for the Potts model at pixel (i, j) with value `val`.
    :param x: numpy.ndarray, current state of the image
    :param i: int, row index of the pixel
    :param j: int, column index of the pixel
    :param val: int, value to evaluate (0 or 1)
    :param beta: float, inverse temperature parameter
    :return: float, local energy contribution
    """
    energy = 0
    for ni, nj in neighbors(i, j, x.shape):
        energy += (val != x[ni, nj])
    return beta * energy

def gibbs_sampler_step(x, y, sigma, beta):
    """
    Perform one step of Gibbs sampling for the Potts model.
    :param x: numpy.ndarray, current state of the image
    :param y: numpy.ndarray, observed noisy image
    :param sigma: float, standard deviation of the noise
    :param beta: float, inverse temperature parameter
    :return: numpy.ndarray, updated state of the image
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            probs = []
            for val in [0, 1]:
                data_term = (y[i, j] - val) ** 2 / (2 * sigma**2)
                prior_term = potts_local_energy(x, i, j, val, beta)
                energy = data_term + prior_term
                probs.append(np.exp(-energy))
            probs = np.array(probs)
            probs /= np.sum(probs)
            x[i, j] = np.random.choice([0, 1], p=probs)
    return x

def simulated_annealing_with_samples(y, n_iter=50, sigma=0.1, beta_init=0.5, cooling=0.95):
    """
    Perform simulated annealing using Gibbs sampling to denoise an image and collect samples.
    :param y: numpy.ndarray, observed noisy image
    :param n_iter: int, number of iterations for Gibbs sampling
    :param sigma: float, standard deviation of the noise
    :param beta_init: float, initial inverse temperature parameter
    :param cooling: float, cooling factor for the inverse temperature
    :return: tuple, (denoised image, list of samples)
    """
    x = np.round(y.copy())
    beta = beta_init
    samples = []
    for t in range(n_iter):
        x = gibbs_sampler_step(x, y, sigma, beta)
        beta *= cooling
        if t >= n_iter // 2:  # collect samples after burn-in
            samples.append(x.copy())
    return x, samples

if __name__ == '__main__':
    path = 'image.png'  # change this to your image path
    original = load_grayscale_image(path)
    cropped = original[:64, :64]  # work on a small patch for speed
    noisy = add_noise(cropped, sigma=0.3)

    denoised_potts, potts_samples = simulated_annealing_with_samples(noisy, n_iter=40, sigma=0.3, beta_init=1.0, cooling=0.97)
    denoised_quad = simulated_annealing_quadratic(noisy, n_iter=40, sigma=0.3, lam=2.0, alpha=1.0, cooling=0.97)

    map_result = map_estimate(denoised_potts)
    mms_result = mms_estimate(potts_samples)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 5, 1)
    plt.title("Original")
    plt.imshow(cropped, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.title("Noisy")
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.title("Denoised (Potts)")
    plt.imshow(denoised_potts, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title("Denoised (Quadratic)")
    plt.imshow(denoised_quad, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title("MMS Estimate")
    plt.imshow(mms_result, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
