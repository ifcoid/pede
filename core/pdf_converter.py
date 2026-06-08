"""
PDF to Markdown Converter
Menggunakan pymupdf4llm untuk konversi PDF artikel ilmiah ke Markdown
dengan preservasi struktur heading, tabel, dan gambar.
"""

import pymupdf4llm
import fitz  # PyMuPDF
import os
import logging
import re
import numpy as np
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

def clean_markdown_text(md_text: str) -> str:
    """
    Modern RAG Text Cleaner:
    1. Removes isolated page numbers and journal headers/footers (Page boundary stitching).
    2. Fixes hyphenation at the end of lines.
    3. Stitches broken sentences across newlines.
    4. Cleans up broken table artifacts.
    """
    # 1. Remove isolated page numbers (e.g. \n\n 4 \n\n)
    md_text = re.sub(r'\n+\s*\d+\s*\n+', '\n\n', md_text)
    
    # Remove typical journal headers (e.g. > _U. Nawaz et al..._)
    md_text = re.sub(r'\n+\s*> _[^_]+_\s*\n+', '\n\n', md_text)
    
    # 2. Fix Hyphenation (e.g. "super-\nresolution" -> "superresolution")
    md_text = re.sub(r'([a-zA-Z]+)-\s*\n\s*([a-zA-Z]+)', r'\1\2', md_text)
    
    # 3. Stitch broken sentences (lowercase letter or comma followed by newline and lowercase letter)
    md_text = re.sub(r'([a-z,])\s*\n{2,}\s*([a-z])', r'\1 \2', md_text)
    
    # 4. Clean up weird table artifacts (Modern RAG fallback for tables)
    # Replaces empty multi-columns ||| with | and removes weird <br> inside tables
    md_text = re.sub(r'\|<br>\s*', '| ', md_text)
    md_text = re.sub(r'\|\|+', '|', md_text)
    
    return md_text


def fallback_ocr_pdf(pdf_path: str) -> str:
    """Fallback OCR using PaddleOCR for image-based or DRM-protected PDFs."""
    logger.info(f"Initiating OCR fallback for {pdf_path}")
    try:
        from paddleocr import PaddleOCR
        # Initialize lazily to save memory if OCR is not needed
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        images = convert_from_path(pdf_path)
        ocr_text = ""
        for i, img in enumerate(images):
            logger.info(f"OCR processing page {i+1}/{len(images)}...")
            # Convert PIL image to numpy array RGB format for PaddleOCR
            img_np = np.array(img.convert('RGB'))
            
            result = ocr.ocr(img_np, cls=True)
            # PaddleOCR returns a nested list. Sometimes result[0] is None if nothing is found.
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) > 1:
                        text = line[1][0]
                        ocr_text += text + "\n"
            ocr_text += "\n\n"
        return ocr_text
    except Exception as e:
        logger.error(f"OCR Fallback failed: {e}")
        return ""


def convert_pdf_to_markdown(
    pdf_path: str,
    image_dir: str | None = None,
    write_images: bool = True,
) -> str:
    """
    Convert a PDF file to Markdown format.
    
    Args:
        pdf_path: Path to the PDF file
        image_dir: Directory to save extracted images
        write_images: Whether to extract and save images
    
    Returns:
        Markdown string of the document
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    if image_dir and write_images:
        os.makedirs(image_dir, exist_ok=True)
    
    logger.info(f"Converting PDF to markdown: {pdf_path}")
    
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=False,
        write_images=write_images,
        image_path=image_dir or "./data/images",
        show_progress=False,
    )
    
    # Check if we need OCR fallback 
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        page_count = 1

    char_count = len(md_text.strip())
    # Fallback to OCR if less than 1000 chars total OR average chars per page is suspiciously low (< 500)
    # This handles image-based PDFs that have digital text watermarks (e.g., IEEE).
    if char_count < 1000 or (page_count > 0 and (char_count / page_count) < 500):
        logger.warning(f"Extracted text too short ({char_count} chars across {page_count} pages). Attempting OCR fallback...")
        ocr_text = fallback_ocr_pdf(pdf_path)
        if len(ocr_text) > len(md_text):
            md_text = ocr_text
            logger.info(f"OCR fallback successful. New length: {len(md_text)} chars")
    
    # Apply modern RAG text cleaning
    md_text = clean_markdown_text(md_text)
    
    logger.info(f"Conversion complete. Markdown length: {len(md_text)} chars")
    return md_text


def get_pdf_native_metadata(pdf_path: str) -> dict:
    """
    Extract native PDF metadata using PyMuPDF.
    
    Returns dict with keys: title, author, subject, keywords,
    creator, producer, creationDate, modDate, page_count
    """
    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    page_count = len(doc)
    
    # Extract first page text (for heuristic extraction later)
    first_page_text = ""
    if page_count > 0:
        first_page_text = doc[0].get_text("text")
    
    doc.close()
    
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "keywords": meta.get("keywords", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "creation_date": meta.get("creationDate", ""),
        "mod_date": meta.get("modDate", ""),
        "page_count": page_count,
        "first_page_text": first_page_text,
    }
