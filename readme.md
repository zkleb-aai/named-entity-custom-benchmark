# Transcript Analysis and Entity Matching

This repository provides tools for analyzing transcripts, extracting named entities, matching entities between ground truth and predictions, and generating performance statistics.

## Table of Contents
1. [Scripts Overview](#scripts-overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Process Transcript](#1-process-transcript)
   - [Process and Analyze](#2-process-and-analyze)
5. [Environment Variables](#environment-variables)
6. [License](#license)
7. [Contributing](#contributing)
8. [Author](#author)

## Scripts Overview

### 1. `process_transcript.py`
Extracts and organizes named entities from a transcript.

### 2. `process_and_analyze.py`
Matches entities between ground truth and predictions, generating performance statistics.

## Requirements
- Python 3.7+
- Required packages:
  - `requests`
  - `python-dotenv`
  - `fuzzywuzzy`
  - `python-Levenshtein`
  - `jiwer`
  - `whisper-normalizer`
  - `jarowinkler`
  - `fuzzy`

## Installation
Install the required packages with:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Process Transcript
Extract named entities from a transcript and organize them into a timeline.

**Command:**
```bash
python get_entities.py path/to/transcript.txt path/to/output/directory --entity_types NAME ORGANIZATION
```

**Arguments:**
- `transcript_file`: Path to the input transcript file.
- `output_dir`: Directory to save output files.
- `--entity_types`: (Optional) Types of entities to extract (default: NAME, ORGANIZATION).

**Outputs:**
- `entities.json`: Extracted raw entity data.
- `timeline.json`: Entities organized by timeline and position.

### 2. Process and Analyze
Match entities between ground truth and predictions, and generate performance statistics.

**Command:**
```bash
python process_and_analyze.py path/to/ground_truth_timeline.json path/to/ground_truth_transcript.txt path/to/prediction_timeline.json path/to/prediction_transcript.txt path/to/output_folder
```

**Arguments:**
- `ground_truth_timeline`: Path to the ground truth timeline JSON file.
- `ground_truth_transcript`: Path to the ground truth transcript.
- `prediction_timeline`: Path to the prediction timeline JSON file.
- `prediction_transcript`: Path to the prediction transcript.
- `output_folder`: Directory to save the output files.

**Outputs:**
- `matches.json`: Entity matching results.
- `statistics.json`: Performance metrics (e.g., WER, PNER, PNWER).

## Environment Variables
`process_transcript.py` requires a Private AI API key. Set it in your environment or a `.env` file:

```bash
PRIVATE_AI_API_KEY=your_api_key_here
```
