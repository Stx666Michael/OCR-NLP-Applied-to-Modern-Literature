"""
Convert text files of traditional Chinese to simplified Chinese
"""

import os


character_source_traditional = "data/source/CCL6kT.txt"
character_source_simplified = "data/source/CCL6kS.txt"

input_path = "output/text/"
output_path = "output/text_simplified/"

character_list_traditional = []
character_list_simplified = []


if __name__ == "__main__":
    with open(character_source_traditional, "r", encoding='UTF-8') as file:
        # Read all characters at once
        characters = file.read()
        # Convert the string to a list of characters
        character_list_traditional = list(characters)

    with open(character_source_simplified, "r", encoding='UTF-8') as file:
        # Read all characters at once
        characters = file.read()
        # Convert the string to a list of characters
        character_list_simplified = list(characters)

    # Create a dictionary to map traditional characters to simplified characters
    character_map = dict(zip(character_list_traditional, character_list_simplified))

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Convert all text files in the input directory
    for filename in os.listdir(input_path):
        with open(input_path + filename, "r", encoding='UTF-8') as file:
            # Read all text at once
            text = file.read()
            # Convert the text
            converted_text = ''.join([character_map.get(character, character) for character in text])
            # Write the converted text to a new file
            with open(output_path + filename, "w", encoding='UTF-8') as output_file:
                output_file.write(converted_text)
                print("Converted", filename)