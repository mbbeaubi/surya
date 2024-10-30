from paddleocr import PaddleOCR
from paddleocr.paddleocr import parse_lang
import numpy as np
from tqdm import tqdm
from surya.input.processing import slice_bboxes_from_image
from typing import List
from PIL import Image
from paddleocr import PPStructure

ocr = PaddleOCR(lang='en')
struct = PPStructure(lang='en', show_log=True, table=False, ocr=False)

def paddle_detect(images: list):
    ret = []
    for img in tqdm(images, desc="Paddle Detection"):
        result = ocr.ocr(img=np.array(img), det=True, rec=False, cls=False)[0]
        bboxes = []

        if result and len(result) > 0:
            for coords in result:
                bbox = (int(coords[0][0]), int(coords[0][1]), int(coords[2][0]), int(coords[2][1]))
                bboxes.append(bbox)
        
        ret.append(bboxes)
    return ret

def paddle_ocr_all(imgs, bboxes, langs: List[str], ocrs: dict[str, PaddleOCR]):
    paddle_text = []
    for img, bb, lang in tqdm(zip(imgs, bboxes, langs), total=len(langs), desc="Paddle"):
        paddle_text.append(paddle_ocr_single(img, bb, ocrs[lang]))

    return paddle_text


def paddle_ocr_single(img, bboxes, ocr: PaddleOCR):
    line_imgs = slice_bboxes_from_image(img, bboxes)
    
    line_ars = [np.array(line_img) for line_img in line_imgs]
    result = ocr.ocr(line_ars, det=False, rec=True, cls=False)[0]
    
    if result and len(result) > 0:
        return [r[0] for r in result]
    
    return []

def paddle_layout_all(imgs: list[Image.Image]):
    results = []
    for img in tqdm(imgs, desc="layout"):
        ar = np.array(img)
        result = struct(ar)
        results.append([{'label': item['type'], 'bbox': item['bbox']} for item in result])

    return results
