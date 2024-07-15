import logging
import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pytesseract
import webcolors
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the width and height of an image.

    Args:
    - image_path (str): The path to the image file.

    Returns:
    - tuple: A tuple containing the width and height of the image.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image from {image_path}")

        height, width, _ = image.shape
        logging.info(f"Image dimensions retrieved successfully for {image_path}")
        return width, height
    except Exception as e:
        logging.error(f"An error occurred while getting image dimensions: {e}")
        return None, None

def extract_text_on_image(image_location: str) -> List[str]:
    """
    Extract text written on images using OCR (Optical Character Recognition).

    Args:
    - image_location (str): The path to the image file.

    Returns:
    - List[str]: A list of strings containing the extracted text from the image.
    """
    try:
        image = cv2.imread(image_location)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        im2 = image.copy()
        string_array = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            string_array.append(text)

        clean_array = [str(s).replace("\n", " ").replace("\x0c", "").replace("  ", " ").strip() for s in string_array if str(s).strip()]
        logging.info("Text extracted from image successfully")
        return clean_array
    except Exception as e:
        logging.error(f"An unexpected error occurred while extracting text from image: {e}")
        return []

def closest_colour(requested_colour: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Find the closest color name from the CSS3 color names list for a given RGB color.

    Args:
    - requested_colour (Tuple[int, int, int]): The RGB color tuple for which the closest color name is to be found.

    Returns:
    - Tuple[int, int, int]: The RGB color tuple representing the closest color name.
    """
    try:
        min_colours = {}
        for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(key)
            rd = (r_c - requested_colour[0]) ** 2
            gd = (g_c - requested_colour[1]) ** 2
            bd = (b_c - requested_colour[2]) ** 2
            min_colours[(rd + gd + bd)] = (r_c, g_c, b_c)
        return min_colours[min(min_colours.keys())]
    except Exception as e:
        logging.error(f"An unexpected error occurred while finding the closest color: {e}")
        return (0, 0, 0)

def top_colors(image: Image.Image, n: int) -> pd.Series:
    """
    Determine the dominant colors in an image.

    Args:
    - image (Image.Image): The input image.
    - n (int): The number of dominant colors to retrieve.

    Returns:
    - pd.Series: A pandas Series containing the dominant colors and their percentages in the image.
    """
    try:
        image = image.convert('RGB').resize((300, 300))
        detected_colors = [closest_colour(image.getpixel((x, y))) for x in range(image.width) for y in range(image.height)]
        Series_Colors = pd.Series(detected_colors)
        output = Series_Colors.value_counts() / len(Series_Colors)
        logging.info("Dominant colors determined successfully")
        return output.head(n)
    except Exception as e:
        logging.error(f"An unexpected error occurred while determining dominant colors: {e}")
        return pd.Series({})

def extract_dominant_colors(image_location: str) -> pd.Series:
    """
    Determine the dominant colors in an image.

    Args:
    - image_location (str): The path to the image file.

    Returns:
    - pd.Series: A pandas Series containing the dominant colors and their percentages in the image.
    """
    try:
        img = Image.open(image_location)
        result = top_colors(img, 10)
        logging.info("Dominant colors extracted from image successfully")
        return result
    except Exception as e:
        logging.error(f"An unexpected error occurred while determining dominant colors: {e}")
        return pd.Series({})

def plot_dominant_colors(series: pd.Series) -> None:
    """
    Plot a pie chart showing the composition of dominant colors.

    Args:
    - series (pd.Series): A pandas Series containing the dominant colors and their percentages.
    """
    try:
        plt.figure(figsize=(8, 8))
        colors = [(*rgb, 1) for rgb in (np.array(series.index) / 255)]
        series.plot(kind='pie', colors=colors, autopct='%1.1f%%', startangle=140)
        plt.title('Dominant Colors Composition')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.ylabel('')
        plt.show()
    except Exception as e:
        logging.error(f"An unexpected error occurred while plotting dominant colors: {e}")

def remove_background(image_path: str, output_path: str) -> Image.Image:
    """
    Removes the background from an image and saves the result.

    :param image_path: Path to the input image file.
    :param output_path: Path to save the output image file.
    :return: The image with background removed.
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image file '{image_path}' not found.")

        input_image = Image.open(image_path)
        output_image = remove(input_image)
        output_image.save(output_path)

        logging.info(f"Background removed from image '{image_path}'. Result saved to '{output_path}'.")
        return output_image
    except Exception as e:
        logging.error(f"An error occurred while removing the background from image '{image_path}': {e}")
        return None

def resize_image(image_path: str, target_width: int, target_height: int, output_path: str) -> str:
    """
    Resize an image to fit within target dimensions while maintaining aspect ratio.

    Args:
        image_path (str): The path to the image file.
        target_width (int): The desired width of the resized image.
        target_height (int): The desired height of the resized image.
        output_path (str): Path to save the output image file.

    Returns:
        str: The path to the resized image.

    Raises:
        ValueError: If either the target width or height is non-positive.
        FileNotFoundError: If the image file does not exist.
        Exception: For any other unexpected error.
    """
    try:
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Target width and height must be positive integers.")

        image = Image.open(image_path).convert("RGBA")
        original_width, original_height = image.size
        ratio = min(target_width / original_width, target_height / original_height)
        new_size = (int(original_width * ratio), int(original_height * ratio))
        resized_image = image.resize(new_size, Image.ANTIALIAS)

        if output_path:
            resized_image.save(output_path)

        logging.info(f"Image resized and saved to {output_path}")
        return output_path
    except ValueError as ve:
        logging.error(f"Error in resizing image: {ve}")
        raise ve
    except FileNotFoundError as fnfe:
        logging.error(f"Image file not found: {image_path}")
        raise fnfe
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e

def create_combined_image(background_path: str, elements: List[dict]) -> str:
    """
    Create a combined image based on background and elements' positioning and sizing.

    Args:
    - background_path (str): Path to the background image.
    - elements (List[dict]): A list of dictionaries, each containing 'image_path', 'start_position_x', 'start_position_y', 'target_width', and 'target_height'.

    Returns:
    - str: The path to the combined image.
    """
    try:
        # Load the background image
        background = Image.open(background_path).convert("RGBA")
        
        for element in elements:
            # Load and resize element image
            image_path = element["image_path"]
            target_width, target_height = element["target_width"], element["target_height"]
            resized_image_path = resize_image(image_path, target_width, target_height, image_path)
            resized_image = Image.open(resized_image_path).convert("RGBA")

            # Calculate position to center the image within its segment
            start_position_x, start_position_y = element["start_position_x"], element["start_position_y"]
            offset_x = start_position_x + (target_width - resized_image.size[0]) // 2
            offset_y = start_position_y + (target_height - resized_image.size[1]) // 2

            # Place the resized image on the background
            background.paste(resized_image, (int(offset_x), int(offset_y)), resized_image)

        # Save the combined image
        new_path = background_path.replace(".png", "_combined.png")
        background.save(new_path)

        logging.info(f"Combined image created and saved to {new_path}")
        return new_path

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise e

def add_text_to_image(image_path: str, text: str, text_color: Tuple[int, int, int] = (255, 255, 255),
                      font_path: str = "Pillow/Tests/fonts/FreeMono.ttf", font_size: int = 24,
                      position: Tuple[int, int] = (10, 10), font_weight: str = "normal") -> None:
    """
    Adds text to an image with specified color, font, and position.

    Args:
        image_path (str): Path to the input image file.
        text (str): Text to be added to the image.
        text_color (Tuple[int, int, int]): RGB color tuple for the text (default is white).
        font_path (str): Path to the font file (default is FreeMono.ttf).
        font_size (int): Font size (default is 24).
        position (Tuple[int, int]): Position where the text will be placed on the image (default is top-left corner).
        font_weight (str): Font weight ("normal" or "bold", default is "normal").

    Returns:
        None. The image with added text is saved in place.

    Raises:
        FileNotFoundError: If the font file specified by `font_path` is not found.
        Exception: For any other unexpected errors during text addition.
    """
    try:
        # Open the image
        image = Image.open(image_path).convert("RGBA")
        
        # Ensure the font path exists
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Font file '{font_path}' not found.")

        # Load the font
        font = ImageFont.truetype(font_path, font_size)

        # Create ImageDraw object
        draw = ImageDraw.Draw(image)

        # Adjust position for bold text
        if font_weight == "bold":
            position = (position[0] + 1, position[1])

        # Draw the text on the image
        draw.text(position, text, font=font, fill=text_color)

        # Save the image with text
        image.save(image_path)

        logging.info(f"Text '{text}' added to image '{image_path}' successfully.")

    except FileNotFoundError as fnf_error:
        logging.error(f"Font file '{font_path}' not found: {fnf_error}")
        raise fnf_error

    except Exception as e:
        logging.error(f"An unexpected error occurred while adding text to image '{image_path}': {e}")
        raise e

from typing import List, Tuple
from PIL import Image, ImageDraw

def combine_images_horizontally(image_paths: List[str], separation_space: int = 100, vertical_padding: int = 200,
                                background_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Combines multiple images into a new image, displayed horizontally on a larger background.
    Images are centered horizontally within the background and have vertical padding.

    Args:
        image_paths (List[str]): List of paths to the input image files.
        separation_space (int): Space between images in pixels (default is 100).
        vertical_padding (int): Vertical padding for the top and bottom of the images (default is 200).
        background_color (Tuple[int, int, int]): Background color of the new image as an RGB tuple (default is white).

    Returns:
        PIL.Image.Image: Combined image.

    Raises:
        FileNotFoundError: If any of the image files specified by `image_paths` are not found.
        Exception: For any other unexpected errors during image combination.
    """
    try:
        # Open each image and calculate their widths and heights
        images = []
        total_images_width = 0
        max_height = 0

        for path in image_paths:
            img = Image.open(path)
            images.append(img)
            total_images_width += img.width + separation_space
            max_height = max(max_height, img.height)

        # Calculate the background size
        background_width = total_images_width - separation_space + 2 * vertical_padding
        background_height = max_height + 2 * vertical_padding

        # Create the background image
        background = Image.new('RGB', (background_width, background_height), color=background_color)

        # Calculate the starting x coordinate to center the images horizontally
        x_offset = vertical_padding

        # Paste each image, centered vertically
        for img in images:
            y_offset = (background_height - img.height) // 2
            background.paste(img, (x_offset, y_offset))
            x_offset += img.width + separation_space

        return background

    except FileNotFoundError as fnf_error:
        logging.error(f"One or more image files not found: {fnf_error}")
        raise fnf_error

    except Exception as e:
        logging.error(f"An unexpected error occurred during image combination: {e}")
        raise e
