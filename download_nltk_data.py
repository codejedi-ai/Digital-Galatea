"""Script to download NLTK data during Docker build"""
import nltk
import os

# Set NLTK data directory
nltk_data_dir = '/root/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
required_data = ['punkt', 'vader_lexicon']
for data_name in required_data:
    try:
        nltk.download(data_name, quiet=True)
        print(f'Downloaded {data_name}')
    except Exception as e:
        print(f'Failed to download {data_name}: {e}')

print('NLTK download step completed')

