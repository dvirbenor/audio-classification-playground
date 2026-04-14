"""
Script to build category mapping from AudioSet ontology.

Downloads the AudioSet ontology and creates a mapping from each class
to its high-level category (Music, Speech, or SFX).

Run this script once to generate category_mapping.json.
"""

import csv
import json
import os
from pathlib import Path

import requests


ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "class_labels_indices.csv"
OUTPUT_FILE = SCRIPT_DIR / "category_mapping.json"


def download_ontology() -> list[dict]:
    """Download AudioSet ontology from Google Storage."""
    print(f"Downloading ontology from {ONTOLOGY_URL}...")
    response = requests.get(ONTOLOGY_URL)
    response.raise_for_status()
    return response.json()


def build_parent_lookup(ontology: list[dict]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Build parent lookup and id-to-name mapping from ontology.
    
    Returns:
        child_to_parents: Mapping from class id to list of parent ids
        id_to_name: Mapping from class id to human-readable name
    """
    child_to_parents: dict[str, list[str]] = {}
    id_to_name: dict[str, str] = {}

    for node in ontology:
        id_to_name[node['id']] = node['name']
        for child_id in node.get('child_ids', []):
            if child_id not in child_to_parents:
                child_to_parents[child_id] = []
            child_to_parents[child_id].append(node['id'])

    return child_to_parents, id_to_name


def get_ancestors(class_id: str, child_to_parents: dict[str, list[str]]) -> set[str]:
    """Get all ancestors of a class by traversing the ontology hierarchy."""
    ancestors = set()
    queue = [class_id]
    while queue:
        current = queue.pop(0)
        if current in child_to_parents:
            for parent in child_to_parents[current]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
    return ancestors


def map_to_high_level(class_id: str, child_to_parents: dict[str, list[str]], id_to_name: dict[str, str]) -> str:
    """
    Map a class to its high-level category (Music, Speech, or SFX).
    
    Logic:
    - If class or ancestors contain "Music" → Music
    - Else if class or ancestors contain "Speech" or "Human voice" → Speech
    - Else → SFX (default for sound effects)
    """
    ancestors = get_ancestors(class_id, child_to_parents)
    
    # Include the class itself in the check
    all_ids = ancestors | {class_id}
    ancestor_names = {id_to_name.get(aid, "") for aid in all_ids}

    if 'Music' in ancestor_names:
        return 'Music'
    elif 'Speech' in ancestor_names or 'Human voice' in ancestor_names:
        return 'Speech'
    else:
        return 'SFX'


def load_class_labels() -> list[tuple[int, str, str]]:
    """Load class labels from CSV file."""
    classes = []
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f, delimiter=",")
        lines = list(reader)
    
    for i in range(1, len(lines)):  # Skip header
        index = int(lines[i][0])
        mid = lines[i][1]
        display_name = lines[i][2]
        classes.append((index, mid, display_name))
    
    return classes


def build_category_mapping() -> dict:
    """
    Build the complete category mapping.
    
    Returns a dict with:
        - mid_to_category: Mapping from machine ID to high-level category
        - index_to_category: List where index i contains the category for class i
        - categories: List of high-level category names
    """
    # Download and parse ontology
    ontology = download_ontology()
    child_to_parents, id_to_name = build_parent_lookup(ontology)
    
    # Load class labels
    classes = load_class_labels()
    
    # Build mappings
    mid_to_category = {}
    index_to_category = []
    
    for index, mid, display_name in classes:
        category = map_to_high_level(mid, child_to_parents, id_to_name)
        mid_to_category[mid] = category
        index_to_category.append(category)
        print(f"{index:3d}: {display_name:40s} -> {category}")
    
    # Count categories
    category_counts = {}
    for cat in index_to_category:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} classes")
    
    return {
        "mid_to_category": mid_to_category,
        "index_to_category": index_to_category,
        "categories": ["Music", "Speech", "SFX"],
    }


def main():
    """Build and save the category mapping."""
    mapping = build_category_mapping()
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nSaved category mapping to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
