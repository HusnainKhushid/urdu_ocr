"""
Gemini Image Field Extractor
Extracts numbered fields from an image in both English and Urdu using Google Gemini 2.5 Flash
"""

import google.generativeai as genai
from PIL import Image
import os
import json
import re

# Configure API key from environment variable
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY environment variable not set!")
genai.configure(api_key=API_KEY)


def extract_fields_from_image(image: Image.Image, print_to_terminal: bool = True) -> dict:
    """
    Extract named fields from a PIL Image using Gemini 2.5 Flash.
    
    Args:
        image: PIL Image object
        print_to_terminal: Whether to print output to terminal
        
    Returns:
        Dictionary with extracted fields
    """
    # Structured prompt for clean extraction
    prompt = """Analyze this document image and extract the following information.
Return ONLY a valid JSON object with these exact keys (use null if not found):

{
    "serial_number": "the document serial/reference number",
    "name_urdu": "person's name in Urdu script",
    "name_english": "person's name in English/transliteration", 
    "father_name_urdu": "father's name in Urdu",
    "father_name_english": "father's name in English",
    "cnic": "CNIC/ID number if present",
    "date": "any date found",
    "address_urdu": "address in Urdu",
    "address_english": "address in English",
    "fields": [
        {"number": 1, "label": "field label", "value_urdu": "value in urdu", "value_english": "value in english"},
        {"number": 2, "label": "field label", "value_urdu": "value in urdu", "value_english": "value in english"}
    ]
}

Extract ALL numbered fields from the document into the "fields" array.
Return ONLY the JSON, no markdown, no explanation."""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, image])
        raw_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```(?:json)?\n?', '', raw_text)
            raw_text = re.sub(r'\n?```$', '', raw_text)
        
        # Parse JSON
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback: return raw text as a field
            data = {"raw_response": raw_text, "parse_error": True}
        
        # Print formatted output to terminal
        if print_to_terminal:
            print("\n" + "=" * 60)
            print("📋 EXTRACTED DOCUMENT FIELDS")
            print("=" * 60)
            
            # Main fields
            if data.get("serial_number"):
                print(f"  Serial Number    : {data['serial_number']}")
            if data.get("name_urdu") or data.get("name_english"):
                print(f"  Name (Urdu)      : {data.get('name_urdu', '-')}")
                print(f"  Name (English)   : {data.get('name_english', '-')}")
            if data.get("father_name_urdu") or data.get("father_name_english"):
                print(f"  Father (Urdu)    : {data.get('father_name_urdu', '-')}")
                print(f"  Father (English) : {data.get('father_name_english', '-')}")
            if data.get("cnic"):
                print(f"  CNIC             : {data['cnic']}")
            if data.get("date"):
                print(f"  Date             : {data['date']}")
            if data.get("address_urdu") or data.get("address_english"):
                print(f"  Address (Urdu)   : {data.get('address_urdu', '-')}")
                print(f"  Address (English): {data.get('address_english', '-')}")
            
            # Numbered fields
            if data.get("fields"):
                print("\n  ─── Numbered Fields ───")
                for field in data["fields"]:
                    num = field.get("number", "?")
                    label = field.get("label", "")
                    val_u = field.get("value_urdu", "-")
                    val_e = field.get("value_english", "-")
                    print(f"  [{num}] {label}")
                    print(f"       Urdu: {val_u}")
                    print(f"       Eng:  {val_e}")
            
            print("=" * 60 + "\n")
        
        return data
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            msg = "Gemini API quota exceeded. Please wait or use a new API key."
        else:
            msg = f"Gemini error: {error_msg[:100]}"
        
        if print_to_terminal:
            print(f"\n❌ {msg}\n")
        
        return {"error": msg}


def format_fields_for_display(data: dict) -> str:
    """Format extracted fields as a clean string for UI display."""
    if "error" in data:
        return f"⚠️ {data['error']}"
    
    if data.get("parse_error"):
        return data.get("raw_response", "Could not parse response")
    
    lines = []
    
    # Main fields
    if data.get("serial_number"):
        lines.append(f"📌 Serial Number: {data['serial_number']}")
    
    if data.get("name_urdu") or data.get("name_english"):
        lines.append(f"👤 Name: {data.get('name_urdu', '')}  |  {data.get('name_english', '')}")
    
    if data.get("father_name_urdu") or data.get("father_name_english"):
        lines.append(f"👨 Father: {data.get('father_name_urdu', '')}  |  {data.get('father_name_english', '')}")
    
    if data.get("cnic"):
        lines.append(f"🪪 CNIC: {data['cnic']}")
    
    if data.get("date"):
        lines.append(f"📅 Date: {data['date']}")
    
    if data.get("address_urdu") or data.get("address_english"):
        lines.append(f"📍 Address: {data.get('address_urdu', '')}  |  {data.get('address_english', '')}")
    
    # Numbered fields
    if data.get("fields"):
        lines.append("\n── Numbered Fields ──")
        for field in data["fields"]:
            num = field.get("number", "?")
            label = field.get("label", "Field")
            val_u = field.get("value_urdu", "-")
            val_e = field.get("value_english", "-")
            lines.append(f"[{num}] {label}: {val_u}  |  {val_e}")
    
    return "\n".join(lines) if lines else "No fields extracted"


# CLI interface for standalone usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gemini_extractor.py <image_path>")
        print("Example: python gemini_extractor.py document.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        print(f"\n📷 Loading image: {image_path}")
        image = Image.open(image_path)
        data = extract_fields_from_image(image, print_to_terminal=True)
        
        print("\n📄 Formatted Output:")
        print(format_fields_for_display(data))
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
