import csv

"""Formatting the original captions file"""

input_file = 'dataset/results.csv'
output_file = 'dataset/captions_formatted.txt'

# Open the original file and the output file
with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    reader = csv.reader(f_in, delimiter='|')

    for row in reader:
        # Check if the row has at least 3 columns (image_name, comment_number, comment)
        if len(row) < 3:
            continue  # Skip rows that don’t match the expected format

        # Extract and clean the fields
        image_name = row[0].strip() if row[0] else None
        comment = row[2].strip() if row[2] else None

        # Ensure both image_name and comment are valid (non-empty) strings
        if not image_name or not comment:
            continue

        # Write to the new file in the required format
        f_out.write(f"{image_name}\t{comment}\n")

