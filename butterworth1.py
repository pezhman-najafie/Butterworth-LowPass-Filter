import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_lowpass_filter(image, d0, n):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Compute the center of the image
    center = (image.shape[0] // 2, image.shape[1] // 2)

    # Create a meshgrid of distances from the center
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Compute the Butterworth low-pass filter
    h = 1 / (1 + (np.sqrt(2) - 1) * (d0 / distance) ** (2 * n))

    # Apply the filter to the Fourier transform of the image
    fft_image = np.fft.fft2(gray_image)
    fft_shifted = np.fft.fftshift(fft_image)
    fft_filtered = fft_shifted * h
    filtered_image = np.fft.ifft2(np.fft.ifftshift(fft_filtered)).real

    return filtered_image

# Load Lena image
lena_image = cv2.imread('lena.PNG', cv2.IMREAD_COLOR)

# Apply Butterworth low-pass filter with different parameters
d0_values = [1, 10, 20, 30, 40,50]
n_values = [1, 2, 4, 8]

plt.figure(figsize=(12, 10))

for i, n in enumerate(n_values):
    for j, d0 in enumerate(d0_values):
        # Apply the filter
        filtered_image = butterworth_lowpass_filter(lena_image, d0, n)

        # Plot the filtered image
        plt.subplot(len(n_values), len(d0_values), i * len(d0_values) + j + 1)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(f'n={n}, D0={d0}')
        plt.axis('off')

plt.tight_layout()
plt.show()
