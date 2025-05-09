# Import libraries 
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
from scipy.ndimage import center_of_mass, shift, label

# Select Files
root = Tk()
root.withdraw()

print("Select one or more Raman spectrum map .txt files")
filepaths = filedialog.askopenfilenames(
    title='Select Raman Spectrum Files',
    filetypes=[('Text Files', '*.txt')]
)

if not filepaths:
    raise FileNotFoundError("No files selected. Please try again.")

# Parameters (Peak Locations)
cutoff_low = 0
cutoff_high = 1
color_range = [cutoff_low, cutoff_high]

# Wavenumber windows for C peak and A peak
c1 = 1050
c2 = 1100
a1 = 1770
a2 = 1830

# Define Processing Function
def process_file(filepath):
    print(f"\nProcessing: {filepath}")

    # Load data
    total_data = np.loadtxt(filepath, skiprows=1)
    x_coords = total_data[:, 0]
    y_coords = total_data[:, 1]
    wavenumbers = total_data[:, 2]
    intensities = total_data[:, 3]

    # Find pixel dimensions
    y_pixels = len(np.unique(x_coords))
    x_pixels = len(np.unique(y_coords))

    # Find number of wavenumber points
    for i in range(len(x_coords) - 1):
        if x_coords[i] < x_coords[i + 1]:
            num_wavenumbers = i + 1
            break

    # Fix wavenumber axis (use only unique values)
    wavenumber_axis = wavenumbers[:num_wavenumbers]

    # Build 3D intensity map: [X, Y, Wavenumber]
    intensity_cube = np.zeros((x_pixels, y_pixels, num_wavenumbers))
    for i in range(x_pixels):
        for j in range(y_pixels):
            start_idx = i * y_pixels * num_wavenumbers + j * num_wavenumbers
            end_idx = start_idx + num_wavenumbers
            intensity_cube[i, j, :] = intensities[start_idx:end_idx]

    # Extract baseline spectrum 
    baseline_spectrum = np.mean(intensity_cube[:, 0, :], axis=0)

    # Find peak regions in wavenumber axis
    index_c_min = np.argmin(np.abs(wavenumber_axis - c2))
    index_c_max = np.argmin(np.abs(wavenumber_axis - c1))
    index_a_min = np.argmin(np.abs(wavenumber_axis - a2))
    index_a_max = np.argmin(np.abs(wavenumber_axis - a1))

    # Find baseline peak heights
    c_peak_baseline = np.max(baseline_spectrum[index_c_min:index_c_max])
    a_peak_baseline = np.max(baseline_spectrum[index_a_min:index_a_max])

    # Avoid division by zero
    if c_peak_baseline != 0:
        baseline_ratio = a_peak_baseline / c_peak_baseline
    else:
        baseline_ratio = 0

    # Analyze each pixel
    max_c_peak = np.zeros((x_pixels, y_pixels))
    max_a_peak = np.zeros((x_pixels, y_pixels))

    for i in range(x_pixels):
        for j in range(y_pixels):
            pixel_spectrum = intensity_cube[i, j, :]
            max_c_peak[i, j] = np.max(pixel_spectrum[index_c_min:index_c_max])
            max_a_peak[i, j] = np.max(pixel_spectrum[index_a_min:index_a_max])

    # Calculate A/C ratio for each pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        ac_ratio_map = np.true_divide(max_a_peak, max_c_peak)
        ac_ratio_map[~np.isfinite(ac_ratio_map)] = 0  # replace inf and NaN with 0

    # Subtract baseline ratio
    intensity_map = ac_ratio_map - baseline_ratio
    intensity_map[intensity_map < 0] = 0

    # Noise cleaning
    binary_map = intensity_map > 0.1
    labeled_map, num_features = label(binary_map)

    if num_features > 0:
        largest_region = (labeled_map == np.argmax(np.bincount(labeled_map.ravel())[1:]) + 1)
    else:
        largest_region = np.zeros_like(binary_map)

    cleaned_intensity = intensity_map * largest_region

    # Centering
    cy, cx = center_of_mass(cleaned_intensity)
    center_x, center_y = np.array(cleaned_intensity.shape) / 2
    shift_x = center_x - cx
    shift_y = center_y - cy
    centered_map = shift(cleaned_intensity, shift=(shift_x, shift_y))

    # Plotting
    plt.figure()
    plt.imshow(centered_map, cmap='jet', origin='lower', vmin=color_range[0], vmax=color_range[1])
    plt.title(f"{os.path.basename(filepath)}\nA/C Ratio Map ")
    plt.colorbar(label="Normalized Intensity")
    plt.xlabel("Y Pixels")
    plt.ylabel("X Pixels")
    plt.tight_layout()

# Run the Processing
for filepath in filepaths:
    process_file(filepath)

plt.show()
