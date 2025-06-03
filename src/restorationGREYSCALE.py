import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_grayscale_image(path, how_clipped=64):
    """
    Load an image from the specified path and convert it to grayscale.

    :param path: str, path to the image file
    :param how_clipped: int, size to which the image will be clipped (width and height)
    :return: numpy.ndarray, grayscale image as a 2D array
    """
    image = Image.open(path)
    if image.mode == 'P':
        image = image.convert('RGBA')
    image = image.convert('L')  # 'L' mode is for (8-bit) grayscale
    return np.array(image)[:how_clipped, :how_clipped]

def add_noise(img, sigma=10):
    """
    Add Gaussian noise to a grayscale image.

    :param img: numpy.ndarray, input image
    :param sigma: float, standard deviation of the Gaussian noise
    :return: numpy.ndarray, noisy image clipped to [0, 255]
    """
    noisy_image = img + np.random.normal(0, sigma, img.shape)
    return np.clip(np.round(noisy_image), 0, 255).astype(np.uint8)

def add_noise_only_for_every_n_pixels(img, sigma=10, n=4):
    """
    Add Gaussian noise to every n-th pixel in a grayscale image.

    :param img: 2D numpy.ndarray, grayscale image with values in range [0, 255]
    :param sigma: float, standard deviation of the Gaussian noise
    :param n: int, step interval for selecting pixels (e.g., every n-th pixel)
    :return: 2D numpy.ndarray, image with noise added to every n-th pixel
    """
    noisy_image = img.copy().astype(np.float32)

    # Create a mask that selects every n-th pixel in 1D view
    mask = np.zeros_like(noisy_image, dtype=bool)
    mask.flat[::n] = True

    # Generate and apply noise only at selected positions
    noise = np.random.normal(loc=0, scale=sigma, size=noisy_image.shape)
    noisy_image[mask] += noise[mask]

    # Clip values to valid range and convert back to uint8
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image



def four_neighbors(i, j, shape):
    """
    Yield the coordinates of the 4-connected neighbors (up, down, left, right)
    for the pixel at position (i, j).

    :param i: int, row index
    :param j: int, column index
    :param shape: tuple, shape of the image (height, width)
    :yield: tuple (ni, nj), valid neighboring pixel coordinates
    """
    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
            yield ni, nj

def eight_neighbors(i, j, shape):
    """
    Yield the coordinates of the 8-connected neighbors (including diagonals)
    for the pixel at position (i, j).

    :param i: int, row index
    :param j: int, column index
    :param shape: tuple, shape of the image (height, width)
    :yield: tuple (ni, nj), valid neighboring pixel coordinates
    """
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                yield ni, nj

def potts_local_energy(x, y, i, j, val, sigma, eight_n=False):
    """
    Compute the local energy using the Potts model.

    :param x: numpy.ndarray, current image state
    :param y: numpy.ndarray, noisy observed image
    :param i: int, row index
    :param j: int, column index
    :param val: int, proposed pixel value
    :param sigma: float, noise standard deviation
    :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
    :return: float, energy value
    """
    point_part = ((y[i, j] - val) ** 2) / (2 * sigma**2) # log of probability Y|X
    neighbour_part = sum(val != x[ni, nj] for ni, nj in (eight_neighbors(i, j, x.shape) if eight_n
                                                         else four_neighbors(i, j, x.shape))) # log of prior probability X
    return point_part + neighbour_part

def gibbs_sampler_potts(x, y, sigma, beta, eight_n=False):
    """
    Perform one Gibbs sampling step for the Potts model.

    :param x: numpy.ndarray, current image state
    :param y: numpy.ndarray, noisy image
    :param sigma: float, noise standard deviation
    :param beta: float, inverse temperature parameter
    :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
    :return: numpy.ndarray, updated image
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vals = np.clip(np.array([x[i, j] - 2, x[i, j] - 1, x[i, j], x[i, j] + 1, x[i, j] + 2]), 0, 255)
            probs = [np.exp(-beta * potts_local_energy(x, y, i, j, v, sigma, eight_n=eight_n)) for v in vals]
            probs = np.array(probs)
            probs /= probs.sum()
            print(probs)
            x[i, j] = np.random.choice(vals, p=probs)
    return x

def simulated_annealing_potts(y, n_iter, sigma, beta_init, cooling, eight_n=False):
    """
    Perform simulated annealing with Gibbs sampling (Potts model).

    :param y: numpy.ndarray, noisy image
    :param n_iter: int, number of iterations
    :param sigma: float, noise standard deviation
    :param beta_init: float, initial inverse temperature
    :param cooling: float, cooling factor per iteration
    :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
    :return: tuple (final image, list of collected samples)
    """
    x = y.copy()
    beta = beta_init
    samples = []
    for t in range(n_iter):
        print(f"Potts iteration {t}")
        x = gibbs_sampler_potts(x, y, sigma, beta, eight_n=eight_n)
        beta *= cooling
        if t >= n_iter // 2:
            samples.append(x.copy())
    return x, samples

def quadratic_local_energy(x, y, i, j, val, sigma, lam, alpha, eight_n=False):
    """
    Compute the local energy using a quadratic prior with edge-preserving regularization.

    :param x: numpy.ndarray, current image state
    :param y: numpy.ndarray, noisy image
    :param i: int, row index
    :param j: int, column index
    :param val: int, proposed pixel value
    :param sigma: float, noise standard deviation
    :param lam: float, regularization weight
    :param alpha: float, truncation threshold to preserve edges
    :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
    :return: float, energy value
    """
    point_part = ((y[i, j] - val) ** 2) / (2 * sigma**2) # log of probability Y|X
    neighbour_part = 0 # log of prior probability X
    for ni, nj in (eight_neighbors(i, j, x.shape) if eight_n else four_neighbors(i, j, x.shape)):
        diff = lam * abs(int(val) - int(x[ni, nj]))
        neighbour_part += min(max(diff**2, alpha), 3)
    return point_part + neighbour_part

def gibbs_sampler_quadratic(x, y, sigma, beta, lam, alpha, eight_n=False):
    """
    Perform one Gibbs sampling step using the quadratic model.

    :param x: numpy.ndarray, current image
    :param y: numpy.ndarray, noisy image
    :param sigma: float, noise std deviation
    :param beta: float, inverse temperature
    :param lam: float, regularization weight
    :param alpha: float, edge-preserving threshold
    :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
    :return: numpy.ndarray, updated image
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            vals = np.clip(np.array([x[i, j] - 4, x[i, j] - 3, x[i, j] - 2, x[i, j] - 1, x[i, j], x[i, j] + 1, x[i, j] + 2, x[i, j] + 3, x[i, j] + 4]), 0, 255)
            probs = [np.exp(-beta * quadratic_local_energy(x, y, i, j, v, sigma, lam, alpha, eight_n=eight_n)) for v in vals]
            if all(p == 0 for p in probs):
                #take with bigger value of quadratic_local_energy
                energies = [quadratic_local_energy(x, y, i, j, v, sigma, lam, alpha, eight_n=eight_n) for v in vals]
                min_energy = min(energies)
                probs = [1.0 if energy == min_energy else 0.0 for energy in energies]
            probs = np.array(probs)
            probs /= probs.sum()
            x[i, j] = np.random.choice(vals, p=probs)
    return x

def simulated_annealing_quadratic(y, n_iter, sigma, beta_init, lam, alpha, cooling, eight_n=False):
    """
    Perform simulated annealing using the quadratic model.

    :param y: numpy.ndarray, noisy grayscale image
    :param n_iter: int, number of iterations
    :param sigma: float, noise standard deviation
    :param beta_init: float, initial inverse temperature
    :param lam: float, regularization weight
    :param alpha: float, edge-preserving threshold
    :param cooling: float, temperature decay per iteration
    :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
    :return: tuple (final image, list of samples)
    """
    x = y.copy()
    beta = beta_init
    samples = []
    for t in range(n_iter):
        x = gibbs_sampler_quadratic(x, y, sigma, beta, lam, alpha, eight_n=eight_n)
        beta *= cooling
        if t >= n_iter // 2:
            samples.append(x.copy())
    return x, samples

def map_estimate(x):
    """
    Return the last state of the image as the MAP estimate.

    :param x: numpy.ndarray, final image
    :return: numpy.ndarray, MAP estimate
    """
    return x

def mms_estimate(samples):
    """
    Compute the Mean of the collected samples (MMS estimate).

    :param samples: list of numpy.ndarray, samples from annealing
    :return: numpy.ndarray, mean image
    """
    return np.mean(samples, axis=0).astype(np.uint8)

if __name__ == '__main__':
    path = 'image.png'  # Replace with your image path
    image = load_grayscale_image(path)
    image_noisy = add_noise(image, sigma=10)
    # image_noisy = add_noise_only_for_every_n_pixels(image, sigma=10, n=2)

    denoised_potts, potts_samples = simulated_annealing_potts(
        image_noisy, n_iter=100, sigma=10, beta_init = 0.5, cooling=1.04)

    map_result_potts = map_estimate(denoised_potts)
    mms_result_potts = mms_estimate(potts_samples)

    denoised_quadratic, quadratic_samples = simulated_annealing_quadratic(
        image_noisy, n_iter=20, sigma=10, beta_init = 0.5, lam=0.18, alpha=0.08, cooling=1.1)
    # bigger lambda emphasizes edges more, smaller alpha preserves more edges (to test)
    # alpha influences how much I look at the neighbors, bigger alpha means I look at more neighbors ? (to test)
    # beta is the inverse temperature, bigger beta means less? exploration (to test)

    map_result_quadratic = map_estimate(denoised_quadratic)
    mms_result_quadratic = mms_estimate(quadratic_samples)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 6, 1)
    plt.title("Original")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title("Noisy")
    plt.imshow(image_noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.title("Potts MAP")
    plt.imshow(map_result_potts, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.title("Potts MMS")
    plt.imshow(mms_result_potts, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.title("Quadratic MAP")
    plt.imshow(map_result_quadratic, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title("Quadratic MMS")
    plt.imshow(mms_result_quadratic, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # print mse between noisy and potts map nad mms
    mse_potts_map = np.mean((image - map_result_potts) ** 2)
    mse_potts_mms = np.mean((image - mms_result_potts) ** 2)
    mse_quadratic_map = np.mean((image - map_result_quadratic) ** 2)
    mse_quadratic_mms = np.mean((image - mms_result_quadratic) ** 2)
    print(f"MSE Potts MAP: {mse_potts_map:.2f}",
          f"MSE Potts MMS: {mse_potts_mms:.2f}",
          f"MSE Quadratic MAP: {mse_quadratic_map:.2f}",
          f"MSE Quadratic MMS: {mse_quadratic_mms:.2f}")
    print("Done")

def process_image_quadrantic(image_path, n_iter=20, sigma=10, beta_init=0.5, lam=0.18, alpha=0.08, cooling=1.1,
                             do_not_show=False, how_clipped=64):
    """
    Process an image using the quadratic model for denoising.

    :param image_path: str, path to the input image
    :param n_iter: int, number of iterations for simulated annealing
    :param sigma: float, noise standard deviation
    :param beta_init: float, initial inverse temperature
    :param lam: float, regularization weight
    :param alpha: float, edge-preserving threshold
    :param cooling: float, cooling factor per iteration
    :param do_not_show: bool, if True, do not display the images
    :param how_clipped: int, size to which the image will be clipped (width and height)
    :return: tuple (MAP estimate, MMS estimate, MSE MAP, MSE MMS)
    """
    image_gray = load_grayscale_image(image_path, how_clipped=how_clipped)
    image_gray_noisy = add_noise(image_gray, sigma=sigma)
    denoised_image, samples = simulated_annealing_quadratic(
        image_gray_noisy, n_iter=n_iter, sigma=sigma,
        beta_init=beta_init, lam=lam, alpha=alpha, cooling=cooling)
    map_result = map_estimate(denoised_image)
    mms_result = mms_estimate(samples)
    mse_map = np.mean((image_gray - map_result) ** 2)
    mse_mms = np.mean((image_gray - mms_result) ** 2)
    mse_noisy = np.mean((image_gray - image_gray_noisy) ** 2)
    if not do_not_show:
        title_string = f"MSE NOISY: {mse_noisy:.2f}, MSE MAP: {mse_map:.2f}, MSE MMS: {mse_mms:.2f}"
        plt.figure(figsize=(12, 2))
        plt.suptitle(title_string)
        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(image_gray, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 4, 2)
        plt.title("Noisy")
        plt.imshow(image_gray_noisy, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 4, 3)
        plt.title("MAP Result")
        plt.imshow(map_result, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 4, 4)
        plt.title("MMS Result")
        plt.imshow(mms_result, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return map_result, mms_result, mse_map, mse_mms