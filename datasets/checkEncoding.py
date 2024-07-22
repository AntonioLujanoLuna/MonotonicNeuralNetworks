import os
import chardet

def check_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result['encoding']

def check_dataset_encodings(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            encoding = check_file_encoding(file_path)
            print(f"File: {filename}, Encoding: {encoding}")

if __name__ == "__main__":
    dataset_directory = "."
    check_dataset_encodings(dataset_directory)