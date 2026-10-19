"""
generate_samples.py - Generates realistic synthetic benchmark crop leaf images for testing.
Creates sample leaf images in the 'samples/' directory.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_sample_leaf(output_path: str, leaf_type: str):
    """
    Synthesizes a realistic test leaf image with veins, textures, and disease markings.
    """
    width, height = 400, 400
    # Background: neutral studio/soil/neutral gradient
    img = Image.new("RGB", (width, height), (230, 230, 225))
    draw = ImageDraw.Draw(img)

    if leaf_type == "healthy_corn":
        base_color = (46, 125, 50)
        # Long lanceolate corn leaf
        draw.polygon([(200, 20), (280, 200), (230, 380), (170, 380), (120, 200)], fill=base_color)
        # Center pale vein
        draw.line([(200, 20), (200, 380)], fill=(120, 190, 100), width=6)
        for y in range(40, 370, 15):
            draw.line([(200, y), (200 + int((y % 30) * 2), y + 10)], fill=(70, 150, 70), width=2)
            draw.line([(200, y), (200 - int((y % 30) * 2), y + 10)], fill=(70, 150, 70), width=2)

    elif leaf_type == "tomato_early_blight":
        base_color = (60, 130, 50)
        # Tomato serrated leaf
        draw.ellipse([(100, 80), (300, 320)], fill=base_color)
        draw.polygon([(100, 180), (40, 200), (110, 230)], fill=base_color)
        draw.polygon([(300, 180), (360, 200), (290, 230)], fill=base_color)
        draw.polygon([(200, 60), (200, 20), (220, 70)], fill=base_color)
        # Main vein
        draw.line([(200, 50), (200, 340)], fill=(40, 95, 35), width=4)

        # Concentric ring early blight spots ("bullseye")
        for cx, cy in [(170, 160), (230, 220), (150, 250)]:
            draw.ellipse([(cx - 28, cy - 28), (cx + 28, cy + 28)], fill=(180, 170, 60))
            draw.ellipse([(cx - 20, cy - 20), (cx + 20, cy + 20)], fill=(90, 55, 30))
            draw.ellipse([(cx - 12, cy - 12), (cx + 12, cy + 12)], fill=(130, 85, 45))
            draw.ellipse([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill=(50, 30, 15))

    elif leaf_type == "apple_scab":
        base_color = (55, 120, 45)
        # Oval apple leaf
        draw.ellipse([(90, 60), (310, 340)], fill=base_color)
        draw.polygon([(200, 50), (195, 20), (205, 50)], fill=base_color)
        # Stem
        draw.line([(200, 335), (200, 390)], fill=(110, 80, 50), width=5)
        # Center vein & secondary veins
        draw.line([(200, 45), (200, 340)], fill=(85, 145, 75), width=3)
        for y in range(80, 320, 25):
            draw.line([(200, y), (260, y - 15)], fill=(75, 135, 65), width=2)
            draw.line([(200, y), (140, y - 15)], fill=(75, 135, 65), width=2)

        # Velvety olive-black scab lesions
        for sx, sy in [(160, 140), (220, 180), (180, 240), (240, 260)]:
            draw.ellipse([(sx - 18, sy - 14), (sx + 18, sy + 14)], fill=(40, 45, 30))
            draw.ellipse([(sx - 10, sy - 8), (sx + 10, sy + 8)], fill=(25, 30, 20))

    elif leaf_type == "potato_late_blight":
        base_color = (50, 115, 45)
        # Compound potato leaf
        draw.ellipse([(110, 70), (290, 330)], fill=base_color)
        draw.line([(200, 60), (200, 350)], fill=(75, 135, 70), width=3)
        # Water-soaked irregular greasy dark brown patches
        draw.ellipse([(130, 110), (220, 200)], fill=(65, 50, 35))
        draw.ellipse([(200, 220), (275, 290)], fill=(50, 40, 25))
        # Pale yellow halo margin
        draw.arc([(125, 105), (225, 205)], 0, 360, fill=(160, 150, 70), width=3)

    # Apply subtle blur to blend textures realistically
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img.save(output_path, quality=95)
    print(f"✅ Generated sample: {output_path}")


def main():
    os.makedirs("samples", exist_ok=True)
    samples = [
        ("samples/sample_tomato_early_blight.jpg", "tomato_early_blight"),
        ("samples/sample_apple_scab.jpg", "apple_scab"),
        ("samples/sample_healthy_corn.jpg", "healthy_corn"),
        ("samples/sample_potato_late_blight.jpg", "potato_late_blight")
    ]

    for path, ltype in samples:
        create_sample_leaf(path, ltype)


if __name__ == "__main__":
    main()
