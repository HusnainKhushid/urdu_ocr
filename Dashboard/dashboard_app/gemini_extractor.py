"""
Gemini FIR Document Extractor
Extracts fields from Pakistani Police FIR Form 24.5 in both Urdu and English
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


def extract_fir_fields(image: Image.Image, print_to_terminal: bool = True) -> dict:
    """
    Extract FIR fields from a PIL Image using Gemini 2.5 Flash.
    Specifically designed for Pakistani Police Form 24.5
    """
    
    # Specialized prompt for Pakistani FIR Form 24.5
    prompt = """You are analyzing a Pakistani Police FIR document (First Information Report - پولیس فارم نمبر 5-24).

Extract ALL information and return a valid JSON object with this EXACT structure:

{
    "header": {
        "form_number": "پولیس فارم نمبر value",
        "serial_number": "سیریل نمبر value",
        "fir_number": "FIR number (like 604/23, 61/24)",
        "police_station": "قانونی station name in both Urdu and English",
        "district": "ضلع district name",
        "ps_file_number": "پی ایس فائل نمبر",
        "date_time_occurrence": "تاریخ وقت و قوعہ"
    },
    "fields": [
        {
            "number": 1,
            "label_urdu": "تاریخ و وقت رپورٹ",
            "label_english": "Date & Time of Report",
            "value_urdu": "exact Urdu value from document",
            "value_english": "English translation"
        },
        {
            "number": 2,
            "label_urdu": "نام و سکونت اطلاع دہندہ و مستغیث",
            "label_english": "Name & Residence of Informant/Complainant",
            "value_urdu": "full name, address, CNIC, phone in Urdu",
            "value_english": "English translation with all details"
        },
        {
            "number": 3,
            "label_urdu": "مختصر کیفیت جرم (مع دفعہ) دائل اگر کچھ کھلم کھلا ہے",
            "label_english": "Brief Description of Crime (with Sections)",
            "value_urdu": "جرم type and sections like 380 ت پ, 392 ت پ",
            "value_english": "Crime type and Pakistan Penal Code sections"
        },
        {
            "number": 4,
            "label_urdu": "جائے وقوعہ قصبہ قانونگو سے اور مسافت",
            "label_english": "Place of Occurrence & Distance from PS",
            "value_urdu": "location in Urdu",
            "value_english": "location in English"
        },
        {
            "number": 5,
            "label_urdu": "کاروائی حفاظتی تفتیشی اگر اطلاع درج کرنے میں کچھ توقف ہو",
            "label_english": "Investigation Action / If Delay in Registration",
            "value_urdu": "action taken in Urdu",
            "value_english": "action taken in English"
        },
        {
            "number": 6,
            "label_urdu": "قانونی سے روانگی کی تاریخ وقت",
            "label_english": "Date/Time of Departure from Police Station",
            "value_urdu": "date/time in Urdu",
            "value_english": "date/time in English"
        }
    ],
    "officer": {
        "name": "Officer name",
        "rank": "ASI/SI/SHO/P-ASI etc",
        "badge_number": "ضابط نمبر",
        "phone": "ٹیلی فون نمبر",
        "signature_date": "date signed"
    },
    "complainant": {
        "name_urdu": "مستغیث name in Urdu",
        "name_english": "Complainant name in English",
        "father_name": "والد name",
        "cnic": "CNIC number if present",
        "phone": "phone number",
        "address_urdu": "address in Urdu",
        "address_english": "address in English"
    },
    "crime": {
        "sections": ["380", "392", "147", "149", "302"],
        "type_urdu": "جرم type in Urdu (چوری، ڈکیتی، قتل)",
        "type_english": "Crime type (Theft, Robbery, Murder, etc)",
        "stolen_property": "description of stolen items",
        "value_rupees": "monetary value in rupees"
    },
    "narrative": {
        "urdu": "COMPLETE FIR statement/narrative in Urdu (ابتدائی اطلاع درج کریں section - the detailed paragraph)",
        "english": "COMPLETE English translation of the FIR narrative word by word"
    },
    "accused": {
        "names": ["list of accused names if mentioned"],
        "descriptions": "physical descriptions if any"
    }
}

CRITICAL INSTRUCTIONS:
1. Extract the COMPLETE narrative/statement (the long paragraph at the bottom)
2. Translate EVERYTHING to English accurately
3. Preserve ALL section numbers exactly (380, 392, 147, 149, 302, 324, 506, etc.)
4. Include all phone numbers, CNIC numbers, addresses EXACTLY as written
5. Extract ALL 6 numbered fields from the table
6. Use null for fields not found
7. Return ONLY valid JSON - no markdown, no explanation, no code blocks"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, image])
        raw_text = response.text.strip()
        
        # Clean markdown code blocks if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r'^```(?:json)?\n?', '', raw_text)
            raw_text = re.sub(r'\n?```$', '', raw_text)
        
        # Parse JSON
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            data = {"raw_response": raw_text, "parse_error": True}
        
        # Print to terminal
        if print_to_terminal:
            print_fir_to_terminal(data)
        
        return data
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            msg = "Gemini API quota exceeded. Please wait or use a new API key."
        else:
            msg = f"Gemini error: {error_msg[:200]}"
        
        if print_to_terminal:
            print(f"\n❌ {msg}\n")
        
        return {"error": msg}


def print_fir_to_terminal(data: dict):
    """Print FIR data in formatted way to terminal."""
    print("\n" + "═" * 80)
    print("                         📋 EXTRACTED FIR DOCUMENT")
    print("                    پولیس فارم نمبر 5-24 - Police Form 24.5")
    print("═" * 80)
    
    if data.get("parse_error"):
        print("⚠️ JSON parse error. Raw response:")
        print(data.get("raw_response", "No response")[:1000])
        return
    
    if data.get("error"):
        print(f"❌ Error: {data['error']}")
        return
    
    # Header
    header = data.get("header", {})
    if header:
        print("\n┌─ HEADER / ہیڈر ──────────────────────────────────────────────────────────┐")
        print(f"│ Serial No (سیریل نمبر): {header.get('serial_number', '-')}")
        print(f"│ FIR No (نمبر): {header.get('fir_number', '-')}")
        print(f"│ Police Station (قانونی): {header.get('police_station', '-')}")
        print(f"│ District (ضلع): {header.get('district', '-')}")
        print(f"│ Date/Time (تاریخ وقت): {header.get('date_time_occurrence', '-')}")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    # Numbered Fields
    fields = data.get("fields", [])
    if fields:
        print("\n┌─ FIR FIELDS / خانے ─────────────────────────────────────────────────────┐")
        for field in fields:
            num = field.get("number", "?")
            label_u = field.get("label_urdu", "")
            label_e = field.get("label_english", "")
            val_u = field.get("value_urdu", "-")
            val_e = field.get("value_english", "-")
            
            print(f"│")
            print(f"│ [{num}] {label_e}")
            print(f"│     {label_u}")
            print(f"│     ─────────────────────────────────────────────────────────────────")
            print(f"│     اردو: {val_u[:100]}{'...' if len(str(val_u)) > 100 else ''}")
            print(f"│     ENG:  {val_e[:100]}{'...' if len(str(val_e)) > 100 else ''}")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    # Complainant
    complainant = data.get("complainant", {})
    if complainant:
        print("\n┌─ COMPLAINANT / مستغیث ──────────────────────────────────────────────────┐")
        print(f"│ Name: {complainant.get('name_urdu', '-')} / {complainant.get('name_english', '-')}")
        print(f"│ Father: {complainant.get('father_name', '-')}")
        print(f"│ CNIC: {complainant.get('cnic', '-')}")
        print(f"│ Phone: {complainant.get('phone', '-')}")
        print(f"│ Address: {complainant.get('address_english', complainant.get('address_urdu', '-'))}")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    # Crime Details
    crime = data.get("crime", {})
    if crime:
        print("\n┌─ CRIME DETAILS / جرم کی تفصیلات ────────────────────────────────────────┐")
        sections = crime.get("sections", [])
        if sections:
            print(f"│ PPC Sections: {', '.join(str(s) for s in sections)}")
        print(f"│ Crime Type: {crime.get('type_urdu', '-')} / {crime.get('type_english', '-')}")
        if crime.get("stolen_property"):
            print(f"│ Stolen Property: {crime['stolen_property']}")
        if crime.get("value_rupees"):
            print(f"│ Value: Rs. {crime['value_rupees']}")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    # Accused
    accused = data.get("accused", {})
    if accused and accused.get("names"):
        print("\n┌─ ACCUSED / ملزمان ──────────────────────────────────────────────────────┐")
        for name in accused.get("names", []):
            print(f"│ • {name}")
        if accused.get("descriptions"):
            print(f"│ Description: {accused['descriptions']}")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    # Narrative
    narrative = data.get("narrative", {})
    if narrative:
        print("\n┌─ FIR NARRATIVE / بیان ──────────────────────────────────────────────────┐")
        if narrative.get("urdu"):
            print("│ 【اردو】")
            urdu_text = str(narrative["urdu"])
            for i in range(0, min(len(urdu_text), 500), 70):
                print(f"│   {urdu_text[i:i+70]}")
            if len(urdu_text) > 500:
                print("│   ...")
        print("│")
        if narrative.get("english"):
            print("│ 【ENGLISH】")
            eng_text = str(narrative["english"])
            for i in range(0, min(len(eng_text), 800), 70):
                print(f"│   {eng_text[i:i+70]}")
            if len(eng_text) > 800:
                print("│   ...")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    # Officer
    officer = data.get("officer", {})
    if officer:
        print("\n┌─ RECORDING OFFICER / درج کرنے والا افسر ───────────────────────────────┐")
        print(f"│ Name: {officer.get('name', '-')}")
        print(f"│ Rank: {officer.get('rank', '-')}")
        print(f"│ Badge No: {officer.get('badge_number', '-')}")
        print(f"│ Phone: {officer.get('phone', '-')}")
        print(f"│ Date: {officer.get('signature_date', '-')}")
        print("└──────────────────────────────────────────────────────────────────────────┘")
    
    print("═" * 80 + "\n")


def format_fir_for_display(data: dict) -> str:
    """Format FIR for UI display (legacy compatibility)."""
    if "error" in data:
        return f"⚠️ {data['error']}"
    if data.get("parse_error"):
        return data.get("raw_response", "Could not parse")
    return "FIR extracted successfully"


# Legacy compatibility
def extract_fields_from_image(image: Image.Image, print_to_terminal: bool = True) -> dict:
    return extract_fir_fields(image, print_to_terminal)

def format_fields_for_display(data: dict) -> str:
    return format_fir_for_display(data)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python gemini_extractor.py <fir_image>")
        sys.exit(1)
    
    image = Image.open(sys.argv[1])
    extract_fir_fields(image, print_to_terminal=True)
