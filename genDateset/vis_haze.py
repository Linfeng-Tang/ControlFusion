import os
import cv2
import numpy as np

def syn_haze(img, depth, beta_max=2, beta_min=1, A_max=0.9, A_min=0.6,
             color_max=0.0, color_min=0):
    """
    Simulates haze on an image using its depth map.

    Args:
        img (np.array): The input image (normalized to [0, 1]).
        depth (np.array): The depth map for the image (normalized to [0, 1]).
        beta_max (float): Maximum scattering coefficient.
        beta_min (float): Minimum scattering coefficient.
        A_max (float): Maximum atmospheric light.
        A_min (float): Minimum atmospheric light.
        color_max (float): Maximum color distortion for atmospheric light.
        color_min (float): Minimum color distortion for atmospheric light.

    Returns:
        np.array: The image with simulated haze, clipped to [0, 1].
    """
    # Randomly select a scattering coefficient
    beta = np.random.uniform(beta_min, beta_max)
    
    # Calculate the transmission map 't' from the depth map
    # A blur is applied to the depth map to simulate smoother haze transitions
    # np.minimum ensures the effect isn't overly strong in far regions
    t = np.exp(-np.minimum(1 - cv2.blur(depth, (22, 22)), 0.7) * beta)
    
    # Randomly determine the atmospheric light 'A'
    A_base = np.random.uniform(A_min, A_max)
    A_random_color = np.random.uniform(color_min, color_max, 3)
    A = A_base + A_random_color
    
    # Apply the haze model formula: I(x) * t(x) + A * (1 - t(x))
    hazy_img = img * t + A * (1 - t)
    
    return np.clip(hazy_img, 0, 1)


def process_images_for_haze(hq_file, depth_file, out_file):
    """
    Reads images and their depth maps, adds synthetic haze, and saves the results.

    Args:
        hq_file (str): Path to the directory with high-quality source images.
        depth_file (str): Path to the directory with corresponding depth maps.
        out_file (str): Path to the directory where hazy images will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(out_file, exist_ok=True)
    
    try:
        file_list = os.listdir(hq_file)
        print(f"Found {len(file_list)} images to process.")
    except FileNotFoundError:
        print(f"Error: The input directory '{hq_file}' was not found.")
        return

    for filename in file_list:
        hq_image_path = os.path.join(hq_file, filename)
        depth_image_path = os.path.join(depth_file, filename)
        
        # Check if both the image and its corresponding depth map exist
        if not os.path.exists(depth_image_path):
            print(f"Warning: Depth map for '{filename}' not found. Skipping.")
            continue
            
        print(f"Processing '{filename}'...")

        # Read and normalize the high-quality image
        img = cv2.imread(hq_image_path)
        if img is None:
            print(f"Warning: Could not read image '{filename}'. Skipping.")
            continue
        lq = img.copy() / 255.0

        # Read and normalize the depth map
        depth = cv2.imread(depth_image_path) / 255.0
        if depth is None:
            print(f"Warning: Could not read depth map for '{filename}'. Skipping.")
            continue
        
        # Apply the haze simulation
        lq_hazy = syn_haze(lq, depth)

        # Convert back to 8-bit integer format for saving
        out_image = (lq_hazy * 255.0).clip(0, 255).astype(np.uint8)

        # Save the resulting hazy image
        output_path = os.path.join(out_file, filename)
        cv2.imwrite(output_path, out_image)
        # print(f"Saved hazy image to '{output_path}'")

    print("\nProcessing complete.")


if __name__ == "__main__":
    # --- PLEASE MODIFY THESE PATHS ---
    hq_path = ''
    depth_path = ''
    output_path = ''
    # --------------------------------

    process_images_for_haze(hq_path, depth_path, output_path)
