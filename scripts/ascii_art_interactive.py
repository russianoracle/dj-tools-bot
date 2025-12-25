#!/usr/bin/env python3
"""Interactive ASCII Art Generator with color and real-time controls"""

from PIL import Image
import sys
import os
import time

# Extended ASCII characters for more detail (darkest to brightest)
ASCII_DETAILED = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# Ultra-detailed with Unicode block characters (best quality)
ASCII_ULTRA = "█▓▒░@#W$9876543210?!abc;:+=-,._  "

# Smooth gradient with half-blocks for maximum detail
UNICODE_BLOCKS = "█▉▊▋▌▍▎▏ "

# Combined ultra-detailed set
ASCII_PREMIUM = "█▓▒░■◆●◉◈◇◦∙⊙⊚⊛⊜⊝○◌◍◎●◐◑◒◓◔◕◖◗❂☢☣☯☸♠♣♥♦"

class ColoredASCII:
    """ANSI color codes for terminal"""
    RESET = '\033[0m'

    @staticmethod
    def rgb(r, g, b, char):
        """Return colored character using RGB"""
        return f'\033[38;2;{r};{g};{b}m{char}{ColoredASCII.RESET}'

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')

def resize_image(image, new_width=120, double_res=False):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    aspect_ratio = height / width
    if double_res:
        # For half-block rendering, we need 2x vertical resolution
        new_height = int(new_width * aspect_ratio * 2 * 0.45)
    else:
        new_height = int(new_width * aspect_ratio * 0.45)
    return image.resize((new_width, new_height))

def pixels_to_half_blocks(image):
    """Convert image to half-block characters with doubled vertical resolution.
    Uses ▀▄ characters with separate foreground/background colors for 2x vertical detail.
    """
    width, height = image.size

    # Ensure even height for pairing
    if height % 2 != 0:
        image = image.crop((0, 0, width, height - 1))
        height -= 1

    ascii_art = []

    # Process two rows at a time
    for y in range(0, height, 2):
        line = ""
        for x in range(width):
            # Get top and bottom pixels
            top_pixel = image.getpixel((x, y))
            bottom_pixel = image.getpixel((x, y + 1))

            if len(top_pixel) >= 3 and len(bottom_pixel) >= 3:
                top_r, top_g, top_b = top_pixel[:3]
                bottom_r, bottom_g, bottom_b = bottom_pixel[:3]

                # Calculate brightness for both pixels
                top_brightness = 0.299 * top_r + 0.587 * top_g + 0.114 * top_b
                bottom_brightness = 0.299 * bottom_r + 0.587 * bottom_g + 0.114 * bottom_b

                # Choose block character based on brightness difference
                brightness_diff = abs(top_brightness - bottom_brightness)

                if brightness_diff < 15:
                    # Similar colors - use full block with average color
                    avg_r = (top_r + bottom_r) // 2
                    avg_g = (top_g + bottom_g) // 2
                    avg_b = (top_b + bottom_b) // 2
                    line += f'\033[38;2;{avg_r};{avg_g};{avg_b}m█\033[0m'
                elif top_brightness > bottom_brightness:
                    # Top is brighter - use upper half block ▀
                    # Set foreground to top color, background to bottom color
                    line += f'\033[38;2;{top_r};{top_g};{top_b}m\033[48;2;{bottom_r};{bottom_g};{bottom_b}m▀\033[0m'
                else:
                    # Bottom is brighter - use lower half block ▄
                    line += f'\033[38;2;{bottom_r};{bottom_g};{bottom_b}m\033[48;2;{top_r};{top_g};{top_b}m▄\033[0m'
            else:
                # Grayscale fallback
                top_br = top_pixel if isinstance(top_pixel, int) else top_pixel[0]
                bottom_br = bottom_pixel if isinstance(bottom_pixel, int) else bottom_pixel[0]

                if abs(top_br - bottom_br) < 15:
                    line += '█'
                elif top_br > bottom_br:
                    line += '▀'
                else:
                    line += '▄'

        ascii_art.append(line)

    return "\n".join(ascii_art)

def apply_dithering(image):
    """Apply Floyd-Steinberg dithering for smoother gradients"""
    import numpy as np
    from PIL import Image

    img_array = np.array(image, dtype=float)
    height, width = img_array.shape[:2]

    for y in range(height - 1):
        for x in range(1, width - 1):
            if len(img_array.shape) == 3:  # Color
                old_pixel = img_array[y, x].copy()
                new_pixel = np.round(old_pixel / 32) * 32
                img_array[y, x] = new_pixel
                quant_error = old_pixel - new_pixel

                img_array[y, x + 1] += quant_error * 7 / 16
                img_array[y + 1, x - 1] += quant_error * 3 / 16
                img_array[y + 1, x] += quant_error * 5 / 16
                img_array[y + 1, x + 1] += quant_error * 1 / 16
            else:  # Grayscale
                old_pixel = img_array[y, x]
                new_pixel = np.round(old_pixel / 32) * 32
                img_array[y, x] = new_pixel
                quant_error = old_pixel - new_pixel

                img_array[y, x + 1] += quant_error * 7 / 16
                img_array[y + 1, x - 1] += quant_error * 3 / 16
                img_array[y + 1, x] += quant_error * 5 / 16
                img_array[y + 1, x + 1] += quant_error * 1 / 16

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def pixels_to_colored_ascii(image, use_color=True, detail_level=1, use_dithering=False):
    """Convert image pixels to colored ASCII characters with detail control"""
    # Select character set based on detail level
    if detail_level == 1:
        chars = "@%#*+=-:. "
    elif detail_level == 2:
        chars = "@%#*oahk+=-:. "
    elif detail_level == 3:
        chars = ASCII_DETAILED
    elif detail_level == 4:
        chars = ASCII_ULTRA
    elif detail_level == 5:
        chars = UNICODE_BLOCKS
    else:  # detail_level == 6
        chars = ASCII_PREMIUM

    # Apply dithering for smoother gradients
    if use_dithering and detail_level >= 4:
        image = apply_dithering(image)

    width, height = image.size
    ascii_art = []

    for y in range(height):
        line = ""
        for x in range(width):
            pixel = image.getpixel((x, y))

            if use_color and len(pixel) >= 3:
                # RGB color mode with perceptual brightness
                r, g, b = pixel[:3]
                # Perceptual brightness (human eye sensitivity)
                brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                char_index = brightness * len(chars) // 256
                char = chars[min(char_index, len(chars) - 1)]
                line += ColoredASCII.rgb(r, g, b, char)
            else:
                # Grayscale mode
                if isinstance(pixel, tuple):
                    brightness = pixel[0]
                else:
                    brightness = pixel
                char_index = brightness * len(chars) // 256
                line += chars[min(char_index, len(chars) - 1)]

        ascii_art.append(line)

    return "\n".join(ascii_art)

def draw_frame(image_path, width, use_color, detail_level, show_info=True):
    """Draw a single frame with current settings"""
    clear_screen()

    try:
        image = Image.open(image_path)
        original_size = image.size

        # Level 7 uses photorealistic half-block mode
        if detail_level == 7:
            resized = resize_image(image, width, double_res=True)
            ascii_art = pixels_to_half_blocks(resized)
        else:
            # Standard processing
            resized = resize_image(image, width)
            if not use_color:
                resized = resized.convert("L")
            ascii_art = pixels_to_colored_ascii(resized, use_color, detail_level, use_dithering=True)

        # Display
        print(ascii_art)

        if show_info:
            detail_names = {
                1: "Basic", 2: "Medium", 3: "Detailed",
                4: "Ultra", 5: "Blocks", 6: "Premium",
                7: "REALISTIC 🎨"
            }
            print(f"\n{'='*60}")
            print(f"Original: {original_size[0]}x{original_size[1]} | "
                  f"ASCII: {resized.width}x{resized.height} chars")
            print(f"Color: {'ON' if use_color else 'OFF'} | "
                  f"Detail: {detail_level}/7 ({detail_names.get(detail_level, 'Custom')}) | Width: {width}")
            print(f"{'='*60}")
            print("Controls: [w/s] width ±10 | [d] detail | [c] color | [q] quit | [h] hide info")

        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def draw_animated(image_path, width=120, use_color=True, detail_level=2, duration=10, style='lines'):
    """Draw ASCII art with animation over specified duration"""
    import random

    try:
        image = Image.open(image_path)
        original_size = image.size

        # Level 7 uses photorealistic mode - draw all at once then animate fade
        if detail_level == 7:
            draw_animated_realistic(image_path, width, duration, style)
            return

        # Process image
        resized = resize_image(image, width)
        if not use_color:
            resized = resized.convert("L")

        # Get dimensions
        img_width, img_height = resized.size

        # Select character set
        if detail_level == 1:
            chars = "@%#*+=-:. "
        elif detail_level == 2:
            chars = "@%#*oahk+=-:. "
        elif detail_level == 3:
            chars = ASCII_DETAILED
        elif detail_level == 4:
            chars = ASCII_ULTRA
        elif detail_level == 5:
            chars = UNICODE_BLOCKS
        else:
            chars = ASCII_PREMIUM

        # Apply dithering for high detail levels
        if detail_level >= 4:
            resized = apply_dithering(resized)

        # Pre-calculate all characters and colors
        char_grid = []
        for y in range(img_height):
            row = []
            for x in range(img_width):
                pixel = resized.getpixel((x, y))

                if use_color and len(pixel) >= 3:
                    r, g, b = pixel[:3]
                    # Perceptual brightness
                    brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                    char_index = brightness * len(chars) // 256
                    char = chars[min(char_index, len(chars) - 1)]
                    colored_char = ColoredASCII.rgb(r, g, b, char)
                    row.append(colored_char)
                else:
                    if isinstance(pixel, tuple):
                        brightness = pixel[0]
                    else:
                        brightness = pixel
                    char_index = brightness * len(chars) // 256
                    row.append(chars[min(char_index, len(chars) - 1)])
            char_grid.append(row)

        clear_screen()

        if style == 'lines':
            # Draw line by line from top to bottom
            delay = duration / img_height
            for y in range(img_height):
                # Move cursor to line y
                print(''.join(char_grid[y]))
                sys.stdout.flush()
                time.sleep(delay)

        elif style == 'sketch':
            # Draw character by character, left to right, top to bottom
            total_chars = img_width * img_height
            delay = duration / total_chars

            for y in range(img_height):
                for x in range(img_width):
                    # Move cursor to position (y, x)
                    print(f'\033[{y+1};{x+1}H{char_grid[y][x]}', end='', flush=True)
                    time.sleep(delay)

        elif style == 'random':
            # Draw in random order
            positions = [(y, x) for y in range(img_height) for x in range(img_width)]
            random.shuffle(positions)

            total_chars = len(positions)
            delay = duration / total_chars

            # Print empty grid first
            for y in range(img_height):
                print(' ' * img_width)

            for y, x in positions:
                print(f'\033[{y+1};{x+1}H{char_grid[y][x]}', end='', flush=True)
                time.sleep(delay)

        elif style == 'waves':
            # Draw in waves from left to right
            delay = duration / img_width
            for x in range(img_width):
                for y in range(img_height):
                    print(f'\033[{y+1};{x+1}H{char_grid[y][x]}', end='', flush=True)
                time.sleep(delay)

        # Final info
        print(f'\033[{img_height+2};1H')  # Move cursor below image
        print(f"\n{'='*60}")
        print(f"Animation complete! ({duration}s, style: {style})")
        print(f"Original: {original_size[0]}x{original_size[1]} | ASCII: {img_width}x{img_height}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")

def draw_animated_realistic(image_path, width=180, duration=10, style='lines'):
    """Animated drawing for realistic mode (level 7) with half-blocks"""
    import random

    try:
        image = Image.open(image_path)
        original_size = image.size

        # Process with double resolution
        resized = resize_image(image, width, double_res=True)
        img_width = resized.size[0]
        img_height = resized.size[1]

        # Ensure even height
        if img_height % 2 != 0:
            resized = resized.crop((0, 0, img_width, img_height - 1))
            img_height -= 1

        # Pre-calculate all half-block characters
        char_grid = []
        for y in range(0, img_height, 2):
            row = []
            for x in range(img_width):
                top_pixel = resized.getpixel((x, y))
                bottom_pixel = resized.getpixel((x, y + 1))

                if len(top_pixel) >= 3 and len(bottom_pixel) >= 3:
                    top_r, top_g, top_b = top_pixel[:3]
                    bottom_r, bottom_g, bottom_b = bottom_pixel[:3]

                    top_brightness = 0.299 * top_r + 0.587 * top_g + 0.114 * top_b
                    bottom_brightness = 0.299 * bottom_r + 0.587 * bottom_g + 0.114 * bottom_b
                    brightness_diff = abs(top_brightness - bottom_brightness)

                    if brightness_diff < 15:
                        avg_r = (top_r + bottom_r) // 2
                        avg_g = (top_g + bottom_g) // 2
                        avg_b = (top_b + bottom_b) // 2
                        char = f'\033[38;2;{avg_r};{avg_g};{avg_b}m█\033[0m'
                    elif top_brightness > bottom_brightness:
                        char = f'\033[38;2;{top_r};{top_g};{top_b}m\033[48;2;{bottom_r};{bottom_g};{bottom_b}m▀\033[0m'
                    else:
                        char = f'\033[38;2;{bottom_r};{bottom_g};{bottom_b}m\033[48;2;{top_r};{top_g};{top_b}m▄\033[0m'
                    row.append(char)

            char_grid.append(row)

        clear_screen()
        output_height = len(char_grid)

        if style == 'lines':
            delay = duration / output_height
            for y in range(output_height):
                print(''.join(char_grid[y]))
                sys.stdout.flush()
                time.sleep(delay)

        elif style == 'sketch':
            total_chars = img_width * output_height
            delay = duration / total_chars
            for y in range(output_height):
                for x in range(img_width):
                    print(f'\033[{y+1};{x+1}H{char_grid[y][x]}', end='', flush=True)
                    time.sleep(delay)

        elif style == 'random':
            positions = [(y, x) for y in range(output_height) for x in range(img_width)]
            random.shuffle(positions)
            delay = duration / len(positions)

            for y in range(output_height):
                print(' ' * img_width)

            for y, x in positions:
                print(f'\033[{y+1};{x+1}H{char_grid[y][x]}', end='', flush=True)
                time.sleep(delay)

        elif style == 'waves':
            delay = duration / img_width
            for x in range(img_width):
                for y in range(output_height):
                    print(f'\033[{y+1};{x+1}H{char_grid[y][x]}', end='', flush=True)
                time.sleep(delay)

        print(f'\033[{output_height+2};1H')
        print(f"\n{'='*60}")
        print(f"🎨 REALISTIC MODE Animation complete! ({duration}s, style: {style})")
        print(f"Original: {original_size[0]}x{original_size[1]} | Output: {img_width}x{output_height} half-blocks")
        print(f"Effective resolution: 2x vertical detail")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")

def interactive_mode(image_path):
    """Interactive ASCII art viewer with real-time controls"""
    import select
    import termios
    import tty

    width = 180
    use_color = True
    detail_level = 7  # Start with REALISTIC mode for photorealism
    show_info = True

    # Setup terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        # Initial draw
        draw_frame(image_path, width, use_color, detail_level, show_info)

        print("\nPress any key to start interactive mode...")
        sys.stdin.read(1)

        while True:
            draw_frame(image_path, width, use_color, detail_level, show_info)

            # Non-blocking input check
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()

                if key == 'q':
                    break
                elif key == 'w':
                    width = min(width + 10, 300)
                elif key == 's':
                    width = max(width - 10, 40)
                elif key == 'd':
                    detail_level = (detail_level % 7) + 1
                elif key == 'c':
                    use_color = not use_color
                elif key == 'h':
                    show_info = not show_info
                elif key == '+':
                    width = min(width + 5, 300)
                elif key == '-':
                    width = max(width - 5, 40)

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        clear_screen()
        print("Goodbye!")

def simple_mode(image_path, width=120, use_color=True, detail_level=2):
    """Non-interactive mode - just display once"""
    draw_frame(image_path, width, use_color, detail_level, show_info=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Interactive ASCII Art Generator")
        print("\nUsage:")
        print("  python ascii_art_interactive.py <image_path> [options]")
        print("\nOptions:")
        print("  -i, --interactive    Interactive mode with controls")
        print("  -a, --animate        Animated drawing mode (default: 10s)")
        print("  -t, --time N         Animation duration in seconds (default: 10)")
        print("  -s, --style STYLE    Animation style: lines, sketch, random, waves (default: sketch)")
        print("  -w, --width N        Set width (default: 180)")
        print("  -c, --color          Enable color (default: on)")
        print("  -g, --grayscale      Disable color")
        print("  -d, --detail N       Detail level 1-7 (default: 7)")
        print("\nDetail Levels:")
        print("  1 - Basic      (10 chars: simple ASCII)")
        print("  2 - Medium     (14 chars: extended ASCII)")
        print("  3 - Detailed   (70 chars: full ASCII spectrum)")
        print("  4 - Ultra      (35 chars: Unicode blocks + ASCII)")
        print("  5 - Blocks     (9 chars: smooth Unicode gradient blocks)")
        print("  6 - Premium    (50 chars: geometric Unicode shapes)")
        print("  7 - REALISTIC  (half-blocks: 2x vertical resolution, photorealistic) 🎨⭐")
        print("\nAnimation Styles:")
        print("  lines   - Draw line by line (smooth, top to bottom)")
        print("  sketch  - Draw char by char like artist (left to right)")
        print("  random  - Draw in random order (organic appearance)")
        print("  waves   - Draw in waves left to right (column by column)")
        print("\nExamples:")
        print("  # REALISTIC animated portrait (photorealistic quality)")
        print("  python ascii_art_interactive.py portrait.jpg -a")
        print()
        print("  # Fast realistic animation (5s)")
        print("  python ascii_art_interactive.py portrait.jpg -a -t 5")
        print()
        print("  # Interactive mode (press 'd' to cycle 1-7 levels)")
        print("  python ascii_art_interactive.py portrait.jpg -i")
        print()
        print("  # Static photorealistic render")
        print("  python ascii_art_interactive.py photo.png -d 7 -w 200")
        sys.exit(1)

    image_path = sys.argv[1]

    # Parse arguments
    interactive = '-i' in sys.argv or '--interactive' in sys.argv
    animate = '-a' in sys.argv or '--animate' in sys.argv
    use_color = '-g' not in sys.argv and '--grayscale' not in sys.argv

    width = 180
    if '-w' in sys.argv:
        width = int(sys.argv[sys.argv.index('-w') + 1])
    elif '--width' in sys.argv:
        width = int(sys.argv[sys.argv.index('--width') + 1])

    detail_level = 7  # Default to REALISTIC mode for photorealism
    if '-d' in sys.argv:
        detail_level = int(sys.argv[sys.argv.index('-d') + 1])
    elif '--detail' in sys.argv:
        detail_level = int(sys.argv[sys.argv.index('--detail') + 1])

    duration = 10
    if '-t' in sys.argv:
        duration = float(sys.argv[sys.argv.index('-t') + 1])
    elif '--time' in sys.argv:
        duration = float(sys.argv[sys.argv.index('--time') + 1])

    style = 'sketch'
    if '-s' in sys.argv:
        style = sys.argv[sys.argv.index('-s') + 1]
    elif '--style' in sys.argv:
        style = sys.argv[sys.argv.index('--style') + 1]

    if animate:
        draw_animated(image_path, width, use_color, detail_level, duration, style)
    elif interactive:
        interactive_mode(image_path)
    else:
        simple_mode(image_path, width, use_color, detail_level)
