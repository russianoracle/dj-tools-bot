#!/usr/bin/env python3
"""ASCII Art Generator - converts images to ASCII art in the console"""

from PIL import Image
import sys

# ASCII characters from darkest to brightest
ASCII_CHARS = "@%#*+=-:. "

def resize_image(image, new_width=100):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    aspect_ratio = height / width
    new_height = int(new_width * aspect_ratio * 0.55)  # 0.55 to adjust for character height
    return image.resize((new_width, new_height))

def grayscale(image):
    """Convert image to grayscale"""
    return image.convert("L")

def pixels_to_ascii(image):
    """Map pixels to ASCII characters"""
    pixels = image.getdata()
    ascii_str = ""
    for pixel in pixels:
        ascii_str += ASCII_CHARS[pixel * len(ASCII_CHARS) // 256]
    return ascii_str

def image_to_ascii(image_path, width=100):
    """Convert image to ASCII art"""
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

    image = resize_image(image, width)
    image = grayscale(image)

    ascii_str = pixels_to_ascii(image)
    img_width = image.width

    # Split into lines
    ascii_lines = [ascii_str[i:i+img_width] for i in range(0, len(ascii_str), img_width)]
    return "\n".join(ascii_lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ascii_art.py <image_path> [width]")
        print("Example: python ascii_art.py portrait.jpg 120")
        sys.exit(1)

    image_path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    ascii_art = image_to_ascii(image_path, width)
    if ascii_art:
        print(ascii_art)
