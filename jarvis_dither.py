import os
import sys
import numpy as np
from PIL import Image

def apply_jarvis_dither(image_path, output_path=None):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # 1. Open the image and ensure it's in RGBA format to capture transparency
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size

    # 2. Separate channels: Isolate the alpha mask and convert RGB to grayscale
    r, g, b, a = img.split()
    grayscale = img.convert("L")
    
    # Convert grayscale pixels to a 2D numpy float array for error distribution math
    arr = np.array(grayscale, dtype=float)

    print("Processing Jarvis error-diffusion matrix...")

    # 3. Apply the 48-factor Jarvis, Judice, and Ninke distribution layout
    for y in range(height):
        for x in range(width):
            old_val = arr[y, x]
            new_val = 255.0 if old_val > 127.0 else 0.0
            arr[y, x] = new_val
            err = old_val - new_val

            # Disperse the error fragments down the matrix to neighboring pixels
            if x + 1 < width: arr[y, x+1] += err * 7 / 48
            if x + 2 < width: arr[y, x+2] += err * 5 / 48
            
            if y + 1 < height:
                if x - 2 >= 0:    arr[y+1, x-2] += err * 3 / 48
                if x - 1 >= 0:    arr[y+1, x-1] += err * 5 / 48
                arr[y+1, x]   += err * 7 / 48
                if x + 1 < width:  arr[y+1, x+1] += err * 5 / 48
                if x + 2 < width:  arr[y+1, x+2] += err * 3 / 48
                
            if y + 2 < height:
                if x - 2 >= 0:    arr[y+2, x-2] += err * 1 / 48
                if x - 1 >= 0:    arr[y+2, x-1] += err * 3 / 48
                arr[y+2, x]   += err * 5 / 48
                if x + 1 < width:  arr[y+2, x+1] += err * 3 / 48
                if x + 2 < width:  arr[y+2, x+2] += err * 1 / 48

    # Clip values securely to standard 8-bit limits
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    
    # 4. Rebuild the image: Create the black and white layer, then paste the original alpha back in
    dithered_bw = Image.fromarray(arr, mode="L")
    final_img = Image.merge("RGBA", (dithered_bw, dithered_bw, dithered_bw, a))

    # 5. Save the output file
    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_jarvis{ext}"

    final_img.save(output_path, "PNG")
    print(f"Success! Saved dithered image to: {output_path}")

if __name__ == "__main__":
    # You can drag and drop an image onto the script, or pass it via command line
    if len(sys.argv) < 2:
        print("Usage: python jarvis_dither.py <path_to_image>")
        input("\nPress Enter to exit...")
    else:
        apply_jarvis_dither(sys.argv[1])