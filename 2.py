import cv2
import torch
import numpy as np
from transformers import pipeline
from PIL import Image

def create_point_cloud(color_image, depth_map, focal_length=1000):
    print("Generating 3D point cloud...")
    
    height, width, _ = color_image.shape
    points = []
    
    cx = width / 2
    cy = height / 2

    depth_scale = 1.0 / (depth_map + 1e-6)

    for y in range(height):
        for x in range(width):
            r, g, b = color_image[y, x]
            depth = depth_scale[y, x]
            
            X = (x - cx) * depth / focal_length
            Y = (y - cy) * depth / focal_length
            Z = depth
            
            points.append(f"{X:.4f} {Y:.4f} {Z:.4f} {r} {g} {b}")
            
    return points

def save_ply(points, filename="point_cloud.ply"):
    print(f"Saving point cloud to {filename}...")
    
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
    
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        f.write('\n'.join(points))
    
    print(f"✅ Successfully saved {len(points)} points to {filename}.")

def main():
    try:
        print("Loading depth estimation model (Intel/dpt-hybrid-midas)...")
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"Error loading depth estimation model: {e}")
        return

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
        
        info_text = "Press 'c' to capture, 'q' to quit"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Logitech Webcam', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            print("\n📸 Frame captured!")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            print("Estimating depth...")
            result = depth_estimator(pil_image)
            
            depth_map_pil = result["depth"]
            depth_map_np = np.array(depth_map_pil)

            focal_length_estimate = 1200 
            points = create_point_cloud(frame, depth_map_np, focal_length_estimate)
            save_ply(points, "webcam_capture.ply")
            
            normalized_depth = cv2.normalize(depth_map_np, None, 255, 0, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
            cv2.imshow('Captured Depth Map', colored_depth)
            print("\nDisplaying depth map. Close the windows or press 'q' again to exit.")

    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()