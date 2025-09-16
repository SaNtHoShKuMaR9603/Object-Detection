import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image

def create_point_cloud(color_image, depth_map, focal_length=1000):
    """
    Converts a color image and a depth map into a 3D point cloud.

    Args:
        color_image (np.array): The original color image (used for colorizing points).
        depth_map (np.array): The depth map, where pixel values represent distance.
        focal_length (float): An assumed focal length for the camera. This is a critical
                              parameter that affects the scale and shape of the point cloud.
                              You may need to calibrate your camera to get an accurate value.
    
    Returns:
        list: A list of strings, where each string is a line in the PLY file format.
    """
    print("Generating 3D point cloud...")
    
    # Get the dimensions of the image
    height, width, _ = color_image.shape
    
    # Create a list to hold the vertices for the PLY file
    points = []
    
    # Center of the image (principal point)
    cx = width / 2
    cy = height / 2

    # Invert the depth map for visualization (closer objects are often brighter)
    # We need the raw depth values for calculation, not the normalized one for display
    # This step might need adjustment based on the depth model's output range.
    # The DPT model provides inverse depth, so we can take the reciprocal.
    # We add a small epsilon to avoid division by zero.
    depth_scale = 1.0 / (depth_map + 1e-6)


    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the color of the pixel
            r, g, b = color_image[y, x]
            
            # Get the depth value for this pixel
            depth = depth_scale[y, x]
            
            # Calculate the 3D coordinates
            # This is a simplified pinhole camera model projection
            X = (x - cx) * depth / focal_length
            Y = (y - cy) * depth / focal_length
            Z = depth
            
            # Add the 3D point and its color to our list
            points.append(f"{X:.4f} {Y:.4f} {Z:.4f} {r} {g} {b}")
            
    return points

def save_ply(points, filename="point_cloud.ply"):
    """
    Saves a list of 3D points to a PLY file.

    Args:
        points (list): List of strings, each representing a vertex (X Y Z R G B).
        filename (str): The name of the file to save.
    """
    print(f"Saving point cloud to {filename}...")
    
    # Create the PLY file header
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header"
    ]
    
    # Write the header and points to the file
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        f.write('\n'.join(points))
    
    print(f"✅ Successfully saved {len(points)} points to {filename}.")


def main():
    """Main function to run the depth estimation and point cloud generation."""
    # --- 1. Load the Depth Estimation Model ---
    try:
        print("Loading depth estimation model (Intel/dpt-hybrid-midas)...")
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"Error loading depth estimation model: {e}")
        return

    # --- 2. Initialize the Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n🚀 Starting webcam feed...")
    print("   Press 'c' to capture an image and create a 3D point cloud.")
    print("   Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Display instructions on the frame
        info_text = "Press 'c' to capture, 'q' to quit"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Logitech Webcam', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n📸 Frame captured!")
            
            # --- Process the Captured Frame ---
            # Convert to PIL Image for the model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            print("Estimating depth...")
            result = depth_estimator(pil_image)
            
            # The result is a dictionary containing the depth map as a PIL Image
            # Convert it to a NumPy array for processing
            depth_map_pil = result["depth"]
            depth_map_np = np.array(depth_map_pil)

            # --- Generate and Save Point Cloud ---
            # NOTE: The focal length is a guess. Calibrating your webcam would give a more accurate result.
            # A higher focal length will make the point cloud "flatter", a lower one will exaggerate depth.
            focal_length_estimate = 1200 
            points = create_point_cloud(frame, depth_map_np, focal_length_estimate)
            save_ply(points, "webcam_capture.ply")
            
            # Display the depth map for visualization
            normalized_depth = cv2.normalize(depth_map_np, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
            cv2.imshow('Captured Depth Map', colored_depth)
            print("\nDisplaying depth map. Close the windows or press 'q' again to exit.")

    # --- Clean Up ---
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


'''
***

### How to Use This Script

1.  **Install Libraries:** If you haven't already, install the required packages.
    ```bash
    pip install transformers torch opencv-python Pillow
    ```

2.  **Run the Script:** Execute the code from your terminal.
    ```bash
    python webcam_to_3d.py
    
'''