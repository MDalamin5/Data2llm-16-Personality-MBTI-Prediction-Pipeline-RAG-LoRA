import os
import re

# Input directory containing .txt files
input_dir = "processed/"
# Output directory for cleaned files (create if not exists)
output_dir = "cleaned-txt-data/"
os.makedirs(output_dir, exist_ok=True)

# Function to clean text: normalize spaces, replace multiple newlines with single space, trim
def clean_text(text):
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading/trailing spaces
    text = text.strip()
    return text

# Process all .txt files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read the file content
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        cleaned_content = clean_text(content)
        
        # Write the cleaned content to the output directory
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        print(f"Processed and saved: {output_path}")

print("All .txt files processed and saved to", output_dir)