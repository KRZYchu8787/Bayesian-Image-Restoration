import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_RGB_image(path):
    """
    Load a RGB image and convert it to a numpy array.
    :param path: str, path to the image file
    :return: numpy.ndarray, RGB image
    """
    image = Image.open(path).convert('RGB')  # wczytaj jako RGB
    return np.array(image)

def add_noise(img, sigma=10):
    """
    Add Gaussian noise to an image.
    :param img: numpy.ndarray, input image
    :param sigma: float, standard deviation of the Gaussian noise
    :return: numpy.ndarray, noisy image
    """
    noisy_image = img + np.random.normal(0, sigma, img.shape)
    return np.clip(np.round(noisy_image), 0, 255) # zwracamy zaburzony obraz mający sens w skali RGB

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
        print(x[4, 4])
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
    print(point_part, "point")
    print(neighbour_part, "neighbour")
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
            print(probs)
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
        print(f"Quadratic iteration {t}")
        x = gibbs_sampler_quadratic(x, y, sigma, beta, lam, alpha, eight_n=eight_n)
        beta *= cooling
        if t >= n_iter // 2:
            samples.append(x.copy())
    return x, samples


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


if __name__ == '__main__':
    path = 'iphone.png'  # change this to your image path
    image = load_RGB_image(path)

    # Rozbicie na kanały
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    # work on a small patch for speed
    R_cropped = R[:64, :64]  
    G_cropped = G[:64, :64]
    B_cropped = B[:64, :64]

    # add noise for each channel
    R_noisy = add_noise(R_cropped, sigma=10)
    G_noisy = add_noise(G_cropped, sigma=10)
    B_noisy = add_noise(B_cropped, sigma=10)

    n_iters=23
    # simulated annealing for each channel
    #R_denoised_potts, R_potts_samples = simulated_annealing_with_samples(R_noisy, n_iter=1, sigma=10, beta_init=1.0, cooling=0.97)
    R_denoised_quadratic, R_quadratic_samples = simulated_annealing_quadratic(R_noisy, n_iter=n_iters, beta_init=1.0, sigma=10, lam=0.3, alpha=0.18, cooling=1.2)

    #G_denoised_potts, G_potts_samples = simulated_annealing_with_samples(G_noisy, n_iter=1, sigma=10, beta_init=1.0, cooling=0.97)
    G_denoised_quadratic, G_quadratic_samples = simulated_annealing_quadratic(G_noisy, n_iter=n_iters, beta_init=1.0, sigma=10, lam=0.3, alpha=0.18, cooling=1.2)

    #B_denoised_potts, B_potts_samples = simulated_annealing_with_samples(B_noisy, n_iter=1, sigma=10, beta_init=1.0, cooling=0.97)
    B_denoised_quadratic, B_quadratic_samples = simulated_annealing_quadratic(B_noisy, n_iter=n_iters, beta_init=1.0, sigma=10, lam=0.3, alpha=0.18, cooling=1.2)

    # # MAP and MMS estimates for each channel for Potts
    # R_map_result_potts = map_estimate(R_denoised_potts)
    # R_mms_result_potts = mms_estimate(R_potts_samples)

    # G_map_result_potts = map_estimate(G_denoised_potts)
    # G_mms_result_potts = mms_estimate(G_potts_samples)

    # B_map_result_potts = map_estimate(B_denoised_potts)
    # B_mms_result_potts = mms_estimate(B_potts_samples)

    #MAP and MMS estimates for each channel for quadratic
    R_map_result_quadratic = map_estimate(R_denoised_quadratic)
    R_mms_result_quadratic = mms_estimate(R_quadratic_samples)

    G_map_result_quadratic = map_estimate(G_denoised_quadratic)
    G_mms_result_quadratic = mms_estimate(G_quadratic_samples)

    B_map_result_quadratic = map_estimate(B_denoised_quadratic)
    B_mms_result_quadratic = mms_estimate(B_quadratic_samples)

    # Składanie kanałów z powrotem do RGB obrazów

    # original
    original_rgb = np.stack([R_cropped, G_cropped, B_cropped], axis=2).astype(np.uint8)

    # noisy
    noisy_rgb = np.stack([R_noisy, G_noisy, B_noisy], axis=2).astype(np.uint8)

    # # MAP estimate Potts
    # map_denoised_potts_rgb = np.stack([R_map_result_potts, G_map_result_potts, B_map_result_potts], axis=2).astype(np.uint8)
    # # MMS estimate Potts
    # mms_denoised_potts_rgb = np.stack([R_mms_result_potts, G_mms_result_potts, B_mms_result_potts], axis=2).astype(np.uint8)


    # MAP estimate quadratic
    map_denoised_quadratic_rgb = np.stack([R_map_result_quadratic, G_map_result_quadratic, B_map_result_quadratic], axis=2).astype(np.uint8)
    # MMS estimate quadratic
    mms_denoised_quadratic_rgb = np.stack([R_mms_result_quadratic, G_mms_result_quadratic, B_mms_result_quadratic], axis=2).astype(np.uint8)

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 6, 1)
    plt.title("Original")
    plt.imshow(original_rgb)
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title("Noisy")
    plt.imshow(noisy_rgb)
    plt.axis('off')

    # plt.subplot(1, 6, 3)
    # plt.title("Potts MAP")
    # plt.imshow(map_denoised_potts_rgb)
    # plt.axis('off')

    # plt.subplot(1, 6, 4)
    # plt.title("Potts MMS")
    # plt.imshow(mms_denoised_potts_rgb)
    # plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.title("Quadratic MAP")
    plt.imshow(map_denoised_quadratic_rgb)
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title("Quadratic MMS")
    plt.imshow(mms_denoised_quadratic_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

