"""
Gemini Image Field Extractor
Extracts numbered fields from an image in both English and Urdu using Google Gemini 2.5 Flash
"""

import google.generativeai as genai
from PIL import Image
import sys

# Configure API key
API_KEY = "AIzaSyAwRo9tIiWscidSymTg8HZFma0A3rXGTDc"
genai.configure(api_key=API_KEY)

def extract_fields(image_path: str) -> None:
    """
    Extract numbered fields from an image using Gemini 2.5 Flash.
    
    Args:
        image_path: Path to the image file
    """
    # Load image
    print(f"\n📷 Loading image: {image_path}")
    image = Image.open(image_path)
    
    # Initialize model
    print("🤖 Initializing Gemini 2.5 Flash...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Prompt for extraction
    prompt = """Extract all the numbered fields from this document/image.
    
For each field, provide:
1. The field number
2. The content in Urdu (if present)
3. The content in English (transliteration or translation)

Also extract:
- Name of the person (if present)
- Serial number / ID (if present)
- Any dates or reference numbers

Format the output clearly with labels."""

    print("📤 Sending to Gemini API...\n")
    print("=" * 60)
    
    # Generate content - sending prompt and image together
    response = model.generate_content([prompt, image])
    
    # Print the extracted text
    print("\n📋 EXTRACTED FIELDS:")
    print("=" * 60)
    print(response.text)
    print("=" * 60)
    
    # ========================================
    # EXPLAINING GEMINI RESPONSE FORMAT
    # ========================================
    print("\n\n" + "=" * 60)
    print("📚 GEMINI RESPONSE FORMAT EXPLANATION")
    print("=" * 60)
    
    print("""
The `response` object from Gemini has several attributes:

1. response.text (str)
   - The main text output from the model
   - This is what you typically want for simple use cases
   - Example: "Field 1: نام - Name..."

2. response.candidates (list)
   - List of candidate responses (usually just 1)
   - Each candidate has:
     - candidate.content.parts[0].text  (the actual text)
     - candidate.finish_reason  (why generation stopped)
     - candidate.safety_ratings  (content safety scores)

3. response.prompt_feedback
   - Feedback about the prompt itself
   - Contains safety ratings for your input
   - block_reason if prompt was blocked

4. response.usage_metadata
   - Token usage statistics:
     - prompt_token_count: tokens in your input
     - candidates_token_count: tokens in output  
     - total_token_count: total tokens used
""")

    # Show actual response structure
    print("\n🔍 ACTUAL RESPONSE OBJECT DETAILS:")
    print("-" * 40)
    
    # Candidates info
    if response.candidates:
        candidate = response.candidates[0]
        print(f"  Finish Reason: {candidate.finish_reason}")
        print(f"  Safety Ratings: {len(candidate.safety_ratings)} categories checked")
        for rating in candidate.safety_ratings:
            print(f"    - {rating.category.name}: {rating.probability.name}")
    
    # Usage metadata
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        print(f"\n  Token Usage:")
        print(f"    - Prompt tokens: {response.usage_metadata.prompt_token_count}")
        print(f"    - Response tokens: {response.usage_metadata.candidates_token_count}")
        print(f"    - Total tokens: {response.usage_metadata.total_token_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gemini_extract.py <image_path>")
        print("Example: python gemini_extract.py document.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        extract_fields(image_path)
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
