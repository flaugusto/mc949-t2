import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# Increase default font size
plt.rcParams.update({'font.size': 16})  # Increased from default

# List of image paths and their corresponding labels
image_paths = [
    'images/matches2-1.png',
    'images/matches2-14.png',
    'images/matches14-16.png'
]

# Create a figure with 1 row and 3 columns
fig, axes = plt.subplots(1, 3, figsize=(24, 8))  # Increased figure size

# Load and display each image with its label
for idx, (ax, img_path) in enumerate(zip(axes, image_paths), 1):
    try:
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        
        # Add subfigure label (a), (b), (c) with larger font and better positioning
        ax.set_title(f'({chr(96+idx)})', 
                    loc='left', 
                    fontsize=24,  # Increased font size
                    fontweight='bold',
                    pad=10)  # Add padding around the label
        
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

# Adjust layout with more padding
plt.tight_layout(pad=3.0)  # Increased padding

# Save the figure with high DPI for better quality
output_path = 'images/matches_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved as {output_path}")

# Close the plot to free memory
plt.close()
