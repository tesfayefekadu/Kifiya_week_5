import pandas as pd
import re

def label_entities_in_csv(input_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Define regex patterns for entity recognition
    size_pattern = re.compile(r'(?i)\bSize\b[:\- ]?([0-9, ]+)', re.UNICODE)
    price_pattern = re.compile(r'(?i)\bPrice\b[:\- ]?([0-9, ]+ birr)', re.UNICODE)
    call_pattern = re.compile(r'(?i)\bcall\b[:\- ]?(\+?[0-9 ]+)', re.UNICODE)
    address_pattern = re.compile(r'አድራሻ.*', re.UNICODE)

    # Function to label entities in CoNLL format
    def label_entities(message):
        if isinstance(message, float):  # Handle NaN or float values
            return "O"  # Return a default output for NaN
        
        tokens = message.split()
        labeled_output = []
        
        for token in tokens:
            # Initialize label as O
            label = 'O'
            
            # Check for product
            if size_pattern.search(token):
                label = 'B-Product'
            elif price_pattern.search(token):
                label = 'B-PRICE'
            elif address_pattern.search(token):
                label = 'B-LOC'
                
            # Append token and its label
            labeled_output.append(f"{token} {label}")
        
        return "\n".join(labeled_output)

    # Process the messages and label them
    with open(output_file_path, 'w', encoding='utf-8') as conll_file:
        for index, row in df.iterrows():
            message = row['Message']
            labeled_message = label_entities(message)
            conll_file.write(labeled_message + "\n\n")  # Separate messages with blank lines

    print("Labeling completed and saved to", output_file_path)

# Example usage (this would be in another notebook):
# label_entities_in_csv('C:/path/to/input_file.csv', 'C:/path/to/output_file.conll')
