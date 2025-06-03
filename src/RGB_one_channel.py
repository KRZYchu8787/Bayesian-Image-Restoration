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
    Add Gaussian noise to an RGB image.
    :param img: numpy.ndarray, input RGB image
    :param sigma: float, standard deviation of the Gaussian noise
    :return: numpy.ndarray, noisy image (uint8)
    """
    img = img.astype(np.float32)
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

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

# def potts_local_energy(x, y, i, j, val, sigma, eight_n=False):
#     """
#     Compute the local energy using the Potts model.

#     :param x: numpy.ndarray, current image state
#     :param y: numpy.ndarray, noisy observed image
#     :param i: int, row index
#     :param j: int, column index
#     :param val: int, proposed pixel value
#     :param sigma: float, noise standard deviation
#     :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
#     :return: float, energy value
#     """
#     point_part = ((y[i, j] - val) ** 2) / (2 * sigma**2) # log of probability Y|X
#     neighbour_part = sum(val != x[ni, nj] for ni, nj in (eight_neighbors(i, j, x.shape) if eight_n
#                                                          else four_neighbors(i, j, x.shape))) # log of prior probability X
#     return point_part + neighbour_part

# def gibbs_sampler_potts(x, y, sigma, beta, eight_n=False):
#     """
#     Perform one Gibbs sampling step for the Potts model.

#     :param x: numpy.ndarray, current image state
#     :param y: numpy.ndarray, noisy image
#     :param sigma: float, noise standard deviation
#     :param beta: float, inverse temperature parameter
#     :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
#     :return: numpy.ndarray, updated image
#     """
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             vals = np.clip(np.array([x[i, j] - 2, x[i, j] - 1, x[i, j], x[i, j] + 1, x[i, j] + 2]), 0, 255)
#             probs = [np.exp(-beta * potts_local_energy(x, y, i, j, v, sigma, eight_n=eight_n)) for v in vals]
#             probs = np.array(probs)
#             probs /= probs.sum()
#             x[i, j] = np.random.choice(vals, p=probs)
            
#     return x

# def simulated_annealing_potts(y, n_iter, sigma, beta_init, cooling, eight_n=False):
#     """
#     Perform simulated annealing with Gibbs sampling (Potts model).

#     :param y: numpy.ndarray, noisy image
#     :param n_iter: int, number of iterations
#     :param sigma: float, noise standard deviation
#     :param beta_init: float, initial inverse temperature
#     :param cooling: float, cooling factor per iteration
#     :param eight_n: bool, whether to use 8-connected neighbors (default False for 4-connected)
#     :return: tuple (final image, list of collected samples)
#     """
#     x = y.copy()
#     beta = beta_init
#     samples = []
#     for t in range(n_iter):
#         print(f"Potts iteration {t}")
#         x = gibbs_sampler_potts(x, y, sigma, beta, eight_n=eight_n)
#         print(x[4, 4])
#         beta *= cooling
#         if t >= n_iter // 2:
#             samples.append(x.copy())
#     return x, samples

def quadratic_local_energy(x, y, i, j, val_rgb, sigma, lam, alpha, eight_n=False):
    """
    Local energy for RGB pixel using quadratic prior (edge-preserving).
    val_rgb: RGB tuple/list/ndarray with shape (3,)
    """
    val_rgb = np.array(val_rgb)
    point_part = np.sum((y[i, j] - val_rgb) ** 2) / (2 * sigma**2)

    neighbour_part = 0
    for ni, nj in (eight_neighbors(i, j, x.shape[:2]) if eight_n else four_neighbors(i, j, x.shape[:2])):
        diff_vec = val_rgb - x[ni, nj]
        diff = lam * np.sqrt(np.linalg.norm(diff_vec))
        print(max(diff**2, alpha), "kk")
        neighbour_part += min(max(diff**2, alpha), 30) / 100
    print(point_part, "point")
    print(neighbour_part, "neighbour")
    return point_part + neighbour_part

def gibbs_sampler_quadratic(x, y, sigma, beta, lam, alpha, eight_n=False):
    """
    One Gibbs sampling step on full RGB image.
    """
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            current = x[i, j]
            proposals = []

            # Generate neighborhood proposals around current RGB value
            for dr in range(-1, 2):
                for dg in range(-1, 2):
                    for db in range(-1, 2):
                        rgb = np.clip(current + np.array([dr, dg, db]), 0, 255)
                        proposals.append(rgb)

            proposals = np.array(proposals, dtype=np.uint8)
            probs = [np.exp(-beta * quadratic_local_energy(x, y, i, j, p, sigma, lam, alpha, eight_n)) for p in proposals]
            probs = np.array(probs)
            probs /= probs.sum()
            idx = np.random.choice(len(proposals), p=probs)
            x[i, j] = proposals[idx]

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
    return np.mean(samples, axis=0).astype(np.uint8)


if __name__ == '__main__':
    path = 'iphone.png'  # Replace with your image path
    image = load_RGB_image(path)
    image_noisy = add_noise(image, sigma=10)
    # image_noisy = add_noise_only_for_every_n_pixels(image, sigma=10, n=2)


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

    noisy_rgb = np.stack([R_noisy, G_noisy, B_noisy], axis=2).astype(np.uint8)


    # denoised_potts, potts_samples = simulated_annealing_potts(
    #     image_noisy, n_iter=35, sigma=10, beta_init = 0.5, cooling=1.04)

    # map_result_potts = map_estimate(denoised_potts)
    # mms_result_potts = mms_estimate(potts_samples)

    denoised_quadratic, quadratic_samples = simulated_annealing_quadratic(
        image_noisy, n_iter=1, sigma=10, beta_init = 0.5, lam=0.18, alpha=0.08, cooling=1.1)

    map_result_quadratic = map_estimate(denoised_quadratic)
    mms_result_quadratic = mms_estimate(quadratic_samples)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 6, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.title("Noisy")
    plt.imshow(image_noisy)
    plt.axis('off')

    # plt.subplot(1, 6, 3)
    # plt.title("Potts MAP")
    # plt.imshow(map_result_potts, cmap='gray')
    # plt.axis('off')

    # plt.subplot(1, 6, 4)
    # plt.title("Potts MMS")
    # plt.imshow(mms_result_potts, cmap='gray')
    # plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.title("Quadratic MAP")
    plt.imshow(map_result_quadratic)
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.title("Quadratic MMS")
    plt.imshow(mms_result_quadratic)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # mse_rgb_map = np.mean((image_cropped - map_result_rgb) ** 2)
    # mse_rgb_mms = np.mean((image_cropped - mms_result_rgb) ** 2)
    # print(f"MSE Quadratic RGB MAP: {mse_rgb_map:.2f}")
    # print(f"MSE Quadratic RGB MMS: {mse_rgb_mms:.2f}")
    # print("Done")