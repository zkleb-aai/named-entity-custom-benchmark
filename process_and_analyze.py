import argparse
import os
import json
import logging
from typing import List, Dict
from fuzzywuzzy import fuzz
from collections import defaultdict
from fuzzy import DMetaphone
from jiwer import wer
from whisper_normalizer.english import EnglishTextNormalizer
from jarowinkler import jarowinkler_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Whisper normalizer
normalizer = EnglishTextNormalizer()

def read_json_file(file_path: str) -> Dict:
    """Read JSON file and return its content."""
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_text(text: str) -> str:
    """Normalize text using Whisper normalizer."""
    return normalizer(text)

def match_entities(ground_truth: List[Dict], transcribed: List[Dict], position_tolerance: int = 10):
    matches = []
    unmatched_truth = ground_truth.copy()
    unmatched_transcribed = transcribed.copy()

    dmetaphone = DMetaphone()

    # First pass: Match entities with exact text, close positions, and high sentence similarity
    for trans_entity in transcribed:
        for truth_entity in unmatched_truth:
            if (trans_entity['text'].lower() == truth_entity['text'].lower() and
                abs(trans_entity['position'] - truth_entity['position']) <= position_tolerance):
                sentence_similarity = fuzz.ratio(trans_entity['sentence'], truth_entity['sentence'])
                if sentence_similarity > 80:
                    matches.append({
                        'truth': truth_entity,
                        'transcribed': trans_entity,
                        'score': 100
                    })
                    unmatched_truth.remove(truth_entity)
                    unmatched_transcribed.remove(trans_entity)
                    break

    # Second pass: Matching based on sentence similarity, position, and entity type
    for trans_entity in unmatched_transcribed[:]:
        best_match = None
        best_score = 0
        
        for truth_entity in unmatched_truth:
            if (abs(trans_entity['position'] - truth_entity['position']) <= position_tolerance and
                trans_entity['entity_type'] == truth_entity['entity_type']):
                sentence_similarity = fuzz.ratio(trans_entity['sentence'], truth_entity['sentence'])
                text_similarity = fuzz.ratio(trans_entity['text'].lower(), truth_entity['text'].lower())
                
                trans_phonetic = dmetaphone(trans_entity['text'])
                truth_phonetic = dmetaphone(truth_entity['text'])
                phonetic_similarity = max(fuzz.ratio(trans_phonetic[0], truth_phonetic[0]),
                                          fuzz.ratio(trans_phonetic[1] or '', truth_phonetic[1] or ''))
                
                position_score = 100 - (abs(trans_entity['position'] - truth_entity['position']) * 10)
                score = 0.5 * sentence_similarity + 0.3 * position_score + 0.15 * text_similarity + 0.05 * phonetic_similarity
                
                if score > best_score:
                    best_score = score
                    best_match = truth_entity

        if best_match and best_score > 50:
            matches.append({
                'truth': best_match,
                'transcribed': trans_entity,
                'score': best_score
            })
            unmatched_truth.remove(best_match)
            unmatched_transcribed.remove(trans_entity)

    # Third pass: Try to match remaining entities with relaxed position constraint
    for trans_entity in unmatched_transcribed[:]:
        best_match = None
        best_score = 0
        
        for truth_entity in unmatched_truth:
            if trans_entity['entity_type'] == truth_entity['entity_type']:
                sentence_similarity = fuzz.ratio(trans_entity['sentence'], truth_entity['sentence'])
                text_similarity = fuzz.ratio(trans_entity['text'].lower(), truth_entity['text'].lower())
                
                trans_phonetic = dmetaphone(trans_entity['text'])
                truth_phonetic = dmetaphone(truth_entity['text'])
                phonetic_similarity = max(fuzz.ratio(trans_phonetic[0], truth_phonetic[0]),
                                          fuzz.ratio(trans_phonetic[1] or '', truth_phonetic[1] or ''))
                
                score = 0.6 * sentence_similarity + 0.3 * text_similarity + 0.1 * phonetic_similarity
                
                if score > best_score:
                    best_score = score
                    best_match = truth_entity

        if best_match and best_score > 80:
            matches.append({
                'truth': best_match,
                'transcribed': trans_entity,
                'score': best_score
            })
            unmatched_truth.remove(best_match)
            unmatched_transcribed.remove(trans_entity)

    return matches, unmatched_truth, unmatched_transcribed

def calculate_wer(truth_text: str, transcribed_text: str) -> float:
    """Calculate Word Error Rate (WER) between two normalized texts."""
    normalized_truth = normalize_text(truth_text)
    normalized_transcribed = normalize_text(transcribed_text)
    return wer(normalized_truth, normalized_transcribed)

def calculate_pner(truth_entities: List[str], transcribed_entities: List[str]) -> float:
    """Calculate Proper Noun Error Rate (PNER) using Jaro-Winkler distance."""
    if not truth_entities:
        return 0.0
    
    total_distance = sum(1 - jarowinkler_similarity(truth, trans) 
                         for truth, trans in zip(truth_entities, transcribed_entities))
    return total_distance / len(truth_entities)

def calculate_pnwer(truth_entities: List[str], transcribed_entities: List[str]) -> float:
    """Calculate Proper Noun Word Error Rate (PNWER)."""
    if not truth_entities:
        return 0.0

    substitutions = sum(1 for truth, trans in zip(truth_entities, transcribed_entities) if truth != trans)
    deletions = max(0, len(truth_entities) - len(transcribed_entities))
    insertions = max(0, len(transcribed_entities) - len(truth_entities))

    total_errors = substitutions + deletions + insertions
    return total_errors / len(truth_entities)

def generate_statistics(matches: List[Dict], unmatched_truth: List[Dict], unmatched_transcribed: List[Dict]) -> Dict:
    """Generate statistics based on the matching results."""
    total_matches = len(matches)
    total_unmatched_truth = len(unmatched_truth)
    total_unmatched_transcribed = len(unmatched_transcribed)
    
    total_entities = total_matches + total_unmatched_truth + total_unmatched_transcribed
    
    match_scores = [match['score'] for match in matches]
    avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0
    
    truth_proper_nouns = [match['truth']['text'] for match in matches]
    transcribed_proper_nouns = [match['transcribed']['text'] for match in matches]
    
    pner = calculate_pner(truth_proper_nouns, transcribed_proper_nouns)
    pnwer = calculate_pnwer(truth_proper_nouns, transcribed_proper_nouns)
    
    return {
        "total_entities": total_entities,
        "total_matches": total_matches,
        "total_unmatched_truth": total_unmatched_truth,
        "total_unmatched_transcribed": total_unmatched_transcribed,
        "match_rate": total_matches / total_entities if total_entities > 0 else 0,
        "unmatched_truth_rate": total_unmatched_truth / total_entities if total_entities > 0 else 0,
        "unmatched_transcribed_rate": total_unmatched_transcribed / total_entities if total_entities > 0 else 0,
        "average_match_score": avg_match_score,
        "pner": pner,
        "pnwer": pnwer
    }

def main():
    parser = argparse.ArgumentParser(description='Process transcripts, match entities, and generate statistics')
    parser.add_argument('ground_truth_timeline', help='Path to the ground truth timeline JSON file')
    parser.add_argument('ground_truth_transcript', help='Path to the ground truth transcript file')
    parser.add_argument('prediction_timeline', help='Path to the prediction timeline JSON file')
    parser.add_argument('prediction_transcript', help='Path to the prediction transcript file')
    parser.add_argument('output_folder', help='Path to the output folder')
    
    args = parser.parse_args()
    
    try:
        # Read timeline files
        ground_truth = read_json_file(args.ground_truth_timeline)
        prediction = read_json_file(args.prediction_timeline)

        # Perform matching
        matches, unmatched_truth, unmatched_transcribed = match_entities(ground_truth, prediction)

        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)  # Create the folder
            print(f"Folder created at: {args.output_folder}")
        
        # Save matching results
        matches_file = f"{args.output_folder}/matches.json"
        with open(matches_file, 'w', encoding='utf-8') as f:
            json.dump({
                'matches': matches,
                'unmatched_truth': unmatched_truth,
                'unmatched_transcribed': unmatched_transcribed
            }, f, indent=2)
        logger.info(f"Matching results saved to {matches_file}")

        # Generate statistics
        stats = generate_statistics(matches, unmatched_truth, unmatched_transcribed)

        # Calculate WER
        with open(args.ground_truth_transcript, 'r') as f:
            truth_text = f.read()
        with open(args.prediction_transcript, 'r') as f:
            prediction_text = f.read()
        
        transcript_wer = calculate_wer(truth_text, prediction_text)
        stats['transcript_wer'] = transcript_wer

        # Save statistics
        stats_file = f"{args.output_folder}/statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_file}")

        # Print summary
        print(f"Matched entities: {len(matches)}")
        print(f"Unmatched ground truth entities: {len(unmatched_truth)}")
        print(f"Unmatched predicted entities: {len(unmatched_transcribed)}")
        print(f"Transcript WER: {transcript_wer}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

# python process_and_analyze.py path/to/ground_truth_timeline.json path/to/ground_truth_transcript.txt path/to/prediction_timeline.json path/to/prediction_transcript.txt path/to/output_folder
