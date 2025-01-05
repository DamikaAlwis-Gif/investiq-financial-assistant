import os
import logging
from .doc_func import load_docs_from_json_files, split_docs
from .vector_store import add_to_vector_store, get_vector_store



def get_json_files_list(directory):
    """Retrieve a list of JSON files from the specified directory."""
    try:
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    except FileNotFoundError:
        logging.error(f"Directory {directory} does not exist.")
        return []
    
    
def load_processed_files(processed_files_path):
    """Load the list of processed files from a text file."""
    if os.path.exists(processed_files_path):
        try:
            with open(processed_files_path, 'r') as file:
                # return the names of the processed files
                return file.read().splitlines()
        except Exception as e:
            logging.error(f"Error reading {processed_files_path}: {e}")
    return []


def save_processed_file(processed_files_path, file_name):
    """Save the processed file name to the text file."""
    try:
        with open(processed_files_path, 'a') as file:
            file.write(file_name + '\n')
        logging.info(f"Processed file {file_name} saved to {processed_files_path}")    
    except Exception as e:
        logging.error(f"Error writing to {processed_files_path}: {e}")


def load_file_content_to_vector_store(json_file):
  """Process a single JSON file if it has not been processed yet."""
   
  try:
    # load the Documents from the JSON file
    docs = load_docs_from_json_files(json_file)
    # split the Documents into smaller chunks
    splits = split_docs(docs)
    # add the chunks to the vector store
    vector_store = get_vector_store()
    add_to_vector_store(splits, vector_store)
    logging.info(f"Add {len(splits)} documents from {json_file} to the vector store.")
    print(f"Add {len(splits)} documents from {json_file} to the vector store.")
            
  except Exception as e:
    logging.error(f"Failed to process {json_file}: {e}")

