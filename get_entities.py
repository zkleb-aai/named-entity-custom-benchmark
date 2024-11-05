import argparse
import logging
import json
import requests
import os
from dotenv import load_dotenv
import re
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment
PRIVATE_AI_API_KEY = os.environ.get('PRIVATE_AI_API_KEY')

class EntityOccurrence:
    def __init__(self, text: str, position: int, entity_type: str, entity_key: str, sentence: str):
        self.text = text
        self.position = position
        self.entity_type = entity_type
        self.entity_key = entity_key
        self.sentence = sentence
    
    def __repr__(self):
        return f"{self.text} ({self.entity_type}, pos:{self.position})"

def extract_named_entities(text, desired_types):
    """Extract named entities using Private AI API."""
    logging.info('Making request to Private AI API...')
    
    if not PRIVATE_AI_API_KEY:
        raise ValueError("PRIVATE_AI_API_KEY environment variable not set")
        
    url = "https://api.private-ai.com/community/v3/process/text"

    entity_types = [{"type": "ENABLE", "value": [entity_type]} for entity_type in desired_types]

    payload = {
        "text": [text],
        "link_batch": False,
        "entity_detection": {
            "accuracy": "high",
            "entity_types": entity_types,
            "return_entity": True
        },
        "processed_text": {
            "type": "MARKER",
            "pattern": "[UNIQUE_NUMBERED_ENTITY_TYPE]"
        }
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": PRIVATE_AI_API_KEY
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        logging.info(f"API Response: {data}")
        
        entity_dict = {}
        text_length = len(text)
        words = re.findall(r'\S+', text)
        
        if data and len(data) > 0 and 'entities' in data[0]:
            for entity in data[0]['entities']:
                entity_key = entity.get('processed_text', '')
                entity_type = entity.get('best_label', '')
                if not entity_key or not entity_type:
                    logging.warning(f"Skipping entity due to missing key or type: {entity}")
                    continue
                
                if entity_key not in entity_dict:
                    entity_dict[entity_key] = {
                        'text': entity.get('text', ''),
                        'type': entity_type,
                        'positions': [],
                        'sentences': []
                    }
                
                location = entity.get('location', {})
                start_pos = location.get('stt_idx')
                end_pos = location.get('end_idx')
                
                if start_pos is not None and end_pos is not None:
                    normalized_pos = int((start_pos / text_length) * 100)
                    entity_dict[entity_key]['positions'].append(normalized_pos)
                    
                    entity_start_word = len(re.findall(r'\S+', text[:start_pos]))
                    start_word = max(0, entity_start_word - 10)
                    end_word = min(len(words), entity_start_word + 10)
                    
                    context = ' '.join(words[start_word:end_word])
                    entity_dict[entity_key]['sentences'].append(context)
                else:
                    logging.warning(f"Skipping context extraction for entity due to missing position information: {entity}")
        
        return entity_dict
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 403:
            logger.error('PrivateAI Rate Limited')
        elif response.status_code == 401:
            logger.error('PrivateAI Auth Error')
        raise e
    except Exception as e:
        logger.error(f"Error in API request: {e}")
        raise

def organize_entities_by_position(entities_json: Dict[str, Any]) -> List[EntityOccurrence]:
    """Organize entities by their positions in the transcript."""
    occurrences = []
    
    for entity_key, entity_data in entities_json.items():
        entity_type = entity_data['type']
        
        for position, sentence in zip(entity_data['positions'], entity_data['sentences']):
            occurrence = EntityOccurrence(
                text=entity_data['text'],
                position=position,
                entity_type=entity_type,
                entity_key=entity_key,
                sentence=sentence
            )
            occurrences.append(occurrence)
    
    return sorted(occurrences, key=lambda x: x.position)

def process_transcript(transcript: str, output_dir: str, entity_types: List[str]):
    """Process a transcript and generate entity and timeline files."""
    # Extract entities
    entities = extract_named_entities(transcript, entity_types)
    
    # Save entities to file
    entities_file = os.path.join(output_dir, 'entities.json')
    with open(entities_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, indent=2)
    logger.info(f"Entities saved to {entities_file}")
    
    # Organize entities
    occurrences = organize_entities_by_position(entities)
    
    # Save timeline to file
    timeline_file = os.path.join(output_dir, 'timeline.json')
    timeline_data = [
        {
            'text': occ.text,
            'position': occ.position,
            'entity_type': occ.entity_type,
            'entity_key': occ.entity_key,
            'sentence': occ.sentence
        }
        for occ in occurrences
    ]
    with open(timeline_file, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, indent=2)
    logger.info(f"Timeline saved to {timeline_file}")

def main():
    parser = argparse.ArgumentParser(description='Process transcript and extract entities')
    parser.add_argument('transcript_file', help='Path to the transcript file')
    parser.add_argument('output_dir', help='Directory to store output files')
    parser.add_argument('--entity_types', nargs='+', default=['NAME', 'ORGANIZATION'],
                        help='Entity types to extract (default: NAME ORGANIZATION)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Read transcript file
        with open(args.transcript_file, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Process transcript
        process_transcript(transcript, args.output_dir, args.entity_types)
        
    except Exception as e:
        logger.error(f"Error processing transcript: {e}")
        raise

if __name__ == "__main__":
    main()

# python get_entities.py path/to/transcript.txt path/to/output/directory --entity_types NAME ORGANIZATION
