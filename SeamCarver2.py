import numpy as np
from PIL import Image
import os

# Function to compute energy map
def compute_energy(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    #  RGB format 
    if len(image.shape) == 3:
        gray = (0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]).astype(np.float64)
    else:
        gray = image.astype(np.float64) 
    
    height, width = gray.shape
    # Initialize energy array with zeros
    energy = np.zeros((height, width), dtype=np.float64)
    
    # Calculate energy for each pixel
    for y in range(height):
        for x in range(width):
            # Get neighboring pixel values
            a = gray[y-1, x-1] if y > 0 and x > 0 else 0.0      # Top-left
            b = gray[y-1, x] if y > 0 else 0.0                  # Top
            c = gray[y-1, x+1] if y > 0 and x < width-1 else 0.0  # Top-right
            d = gray[y, x-1] if x > 0 else 0.0                  # Left
            f = gray[y, x+1] if x < width-1 else 0.0            # Right
            g = gray[y+1, x-1] if y < height-1 and x > 0 else 0.0  # Bottom-left
            h = gray[y+1, x] if y < height-1 else 0.0           # Bottom
            i = gray[y+1, x+1] if y < height-1 and x < width-1 else 0.0  # Bottom-right
            
            # Calculate horizontal and vertical energy components
            xenergy = a + 2*d + g - c - 2*f - i    # Horizontal gradient
            yenergy = a + 2*b + c - g - 2*h - i    # Vertical gradient
            # Compute total energy as magnitude of gradient vector
            energy[y, x] = np.sqrt(xenergy**2 + yenergy**2)
    
    return energy

# Save the energy map as an image for visualization 
def save_energy_map(image, output_path):
    energy = compute_energy(image)
    # Normalize energy values to 0-255 range for visualization 
    energy_min, energy_max = energy.min(), energy.max()
    if energy_max == energy_min: 
        energy_normalized = np.zeros_like(energy, dtype=np.uint8)
    else:
        energy_normalized = ((energy - energy_min) / (energy_max - energy_min) * 255).astype(np.uint8)
    # Convert to PIL image and save
    energy_img = Image.fromarray(energy_normalized, mode='L') 
    energy_img.save(output_path)

# Highlight computed seams on the original image i
def highlight_seams(image, seams, output_path):
    if isinstance(image, Image.Image):
        img_with_seams = np.array(image)
    else:
        img_with_seams = image.copy()
    
    # Ensure image is in RGB format
    if len(img_with_seams.shape) == 2:  # Grayscale
        img_with_seams = np.stack([img_with_seams] * 3, axis=-1)  # Convert to RGB
    
    # Mark each pixel in seams with red color
    for seam in seams:
        for y, x in seam:
            img_with_seams[y, x] = [255, 0, 0]  # RGB format: Red
    seams_img = Image.fromarray(img_with_seams, mode='RGB')
    seams_img.save(output_path)

# Dynamic Programming approach to seam carving
def seam_carving_dp(image, num_seams, output_dir):
    # Convert image to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    height, width = img.shape[:2]
    all_seams = []
    
    # Check if num_seams is valid
    if num_seams >= width:
        print(f"Error: Cannot remove {num_seams} seams from an image of width {width}.")
        return None, None
    
    # Remove specified number of seams
    for seam_idx in range(num_seams):
        energy = compute_energy(img)
        # DP table for minimum energy paths
        dp = np.zeros((height, width), dtype=np.float64)
        dp[0, :] = energy[0, :]  # Initialize first row
        parent = np.zeros((height, width), dtype=int)  # Track seam paths
        
        # Build DP table row by row
        for y in range(1, height):
            for x in range(width):
                # Consider three possible previous pixels (left, center, right)
                min_energy = dp[y-1, x]
                parent[y, x] = x
                if x > 0 and dp[y-1, x-1] < min_energy:
                    min_energy = dp[y-1, x-1]
                    parent[y, x] = x-1
                if x < width-1 and dp[y-1, x+1] < min_energy:
                    min_energy = dp[y-1, x+1]
                    parent[y, x] = x+1
                dp[y, x] = energy[y, x] + min_energy
        
        # Find starting point of minimum energy seam (leftmost if tied)
        seam_end = np.argmin(dp[height-1, :])
        seam = []
        x = seam_end
        # Backtrack to construct the seam
        for y in range(height-1, -1, -1):
            seam.append((y, x))
            x = parent[y, x]
        all_seams.append(seam)
        
        # Create new image with seam removed
        new_img = np.zeros((height, width-1, 3), dtype=np.uint8)
        for y, x in seam:
            new_img[y, :x] = img[y, :x]
            new_img[y, x:] = img[y, x+1:]
        img = new_img
        width -= 1
        
        # Save intermediate result after removing half the seams 
        if seam_idx == num_seams // 2 - 1:
            intermediate_path = os.path.join(output_dir, 'intermediate_dp.jpg')
            intermediate_img = Image.fromarray(img, mode='RGB')
            intermediate_img.save(intermediate_path)
    
    return img, all_seams

# Greedy approach to seam carving
def seam_carving_greedy(image, num_seams, output_dir):
    # Convert image to numpy array if it's a PIL image
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image.copy()
    
    height, width = img.shape[:2]
    all_seams = []
    
    # Check if num_seams is valid
    if num_seams >= width:
        print(f"Error: Cannot remove {num_seams} seams from an image of width {width}.")
        return None, None
    
    # Remove specified number of seams
    for seam_idx in range(num_seams):
        energy = compute_energy(img)
        seam = []
        # Greedily choose minimum energy pixel at each row
        for y in range(height):
            if y == 0:
                x = np.argmin(energy[y, :])  # Start with minimum in first row
            else:
                prev_x = seam[-1][1]
                # Look at three adjacent pixels from previous row
                x_range = slice(max(0, prev_x-1), min(width, prev_x+2))
                x = prev_x - 1 + np.argmin(energy[y, x_range])
            seam.append((y, x))
        all_seams.append(seam)
        
        # Create new image with seam removed
        new_img = np.zeros((height, width-1, 3), dtype=np.uint8)
        for y, x in seam:
            new_img[y, :x] = img[y, :x]
            new_img[y, x:] = img[y, x+1:]
        img = new_img
        width -= 1
        
        # Save intermediate result after removing half the seams (Step 4 in Figure 2)
        if seam_idx == num_seams // 2 - 1:
            intermediate_path = os.path.join(output_dir, 'intermediate_greedy.jpg')
            intermediate_img = Image.fromarray(img, mode='RGB')
            intermediate_img.save(intermediate_path)
    
    return img, all_seams

# Main execution block
if __name__ == "__main__":
    # Define input and output paths
    input_path = r"C:\Users\hp\Desktop\311 project\project\project\images\tower.jpg"
    output_dir = r"C:\Users\hp\Desktop\311 project\project\project\output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define paths for various output files
    energy_map_path = os.path.join(output_dir, 'energy_map.jpg')
    highlighted_seams_dp_path = os.path.join(output_dir, 'highlighted_seams_dp.jpg')
    highlighted_seams_greedy_path = os.path.join(output_dir, 'highlighted_seams_greedy.jpg')
    output_dp_path = os.path.join(output_dir, 'output_dp.jpg')
    output_greedy_path = os.path.join(output_dir, 'output_greedy.jpg')
    
    # Load the input image using PIL (Step 1 in Figure 2)
    try:
        input_image = Image.open(input_path)
        input_image = input_image.convert('RGB')  # Ensure RGB format
    except Exception as e:
        print(f"Error: Could not load image from {input_path}. Exception: {e}")
        exit()
    
    num_seams = 20  # Number of seams to remove
    
    # Generate and save energy map (Step 2 in Figure 2)
    save_energy_map(input_image, energy_map_path)
    
    # Apply DP seam carving and save results
    dp_result, dp_seams = seam_carving_dp(input_image, num_seams, output_dir)
    if dp_result is not None:
        highlight_seams(input_image, dp_seams, highlighted_seams_dp_path)  # Step 3
        # Save final DP result (Step 5)
        dp_img = Image.fromarray(dp_result, mode='RGB')
        dp_img.save(output_dp_path)
    
    # Apply greedy seam carving and save results
    greedy_result, greedy_seams = seam_carving_greedy(input_image, num_seams, output_dir)
    if greedy_result is not None:
        highlight_seams(input_image, greedy_seams, highlighted_seams_greedy_path)  # Step 3
        # Save final greedy result (Step 5)
        greedy_img = Image.fromarray(greedy_result, mode='RGB')
        greedy_img.save(output_greedy_path)
    
    # Print completion message
    print(f"Seam carving completed for 'tower.jpg'. Outputs saved in {output_dir}")