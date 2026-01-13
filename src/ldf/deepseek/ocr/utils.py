# Portions of this code (regex matching and image logic) are adapted from the DeepSeek-OCR repository:
# https://github.com/deepseek-ai/DeepSeek-OCR
# Please refer to the original repository for license and attribution.

import os
import io
import re
from PIL import Image
import fitz

def pdf_to_images(pdf_path, dpi=144):
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            Image.MAX_IMAGE_PIXELS = None
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        pdf_document.close()
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    return images

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    matches_image = []
    matches_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            matches_image.append(a_match[0])
        else:
            matches_other.append(a_match[0])
    return matches, matches_image, matches_other

def crop_and_save_images(image, matches_images, output_dir, page_idx, base_filename):
    image_width, image_height = image.size
    saved_images = []
    
    img_idx = 0
    for match in matches_images:
        try:
            # Simple manual parse for safety
            det_content_match = re.search(r'<\|det\|>(.*?)<\|/det\|>', match)
            if det_content_match:
                cor_list = eval(det_content_match.group(1))
                
                for points in cor_list:
                    x1, y1, x2, y2 = points
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)
                    
                    cropped = image.crop((x1, y1, x2, y2))
                    fname = f"{base_filename}_p{page_idx}_{img_idx}.jpg"
                    save_path = os.path.join(output_dir, fname)
                    cropped.save(save_path)
                    saved_images.append(save_path)
                    img_idx += 1
        except Exception as e:
            print(f"Error processing image crop: {e}")
            continue
            
    return saved_images
