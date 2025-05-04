import torch

from PIL import Image

import os
import json
from datetime import datetime
from pathlib import Path
import math
from transformers import CLIPProcessor, CLIPModel  # Added for CLIP
import sqlite3  # Added for SQLite
import spacy  # Added for NER


# --- Global Candidate Tags for CLIP ---
CANDIDATE_IMAGE_TAGS = [
    "architectural drawing",
    "building exterior",
    "interior design",
    "urban landscape",
    "parametric model",
    "structural detail",
    "conceptual sketch",
    "photorealistic rendering",
    "abstract art",
    "nature photo",
    "object photography",
    "site plan",
    "facade study",
    "material sample",
    "construction detail",
]

# --- Global Settings ---
DB_FILE = "data/tags.sqlite"  # SQLite database file path


def get_closest_color_name(rgb):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "orange": (255, 165, 0),
        "brown": (165, 42, 42),
        "pink": (255, 192, 203),
        "gray": (128, 128, 128),
    }

    r, g, b = rgb
    distances = {}
    for color_name, color_rgb in colors.items():
        distance = math.sqrt(
            (r - color_rgb[0]) ** 2 + (g - color_rgb[1]) ** 2 + (b - color_rgb[2]) ** 2
        )
        distances[color_name] = distance

    return min(distances, key=distances.get)


def get_dominant_color(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize((150, 150))
            paletted = img.quantize(colors=1)
            palette = paletted.getpalette()
            dominant_rgb = (palette[0], palette[1], palette[2])
            color_name = get_closest_color_name(dominant_rgb)
            return color_name

    except Exception as e:
        print(f"Error getting color from {image_path}: {e}")
        return "unknown"


# --- New CLIP Setup ---
def setup_clip_model(model_name="openai/clip-vit-base-patch32"):
    """Loads the CLIP model and processor."""
    try:
        # Set proxy environment variables before loading model if needed
        # You might want to move this to the main execution block or handle proxy setup more globally
        # os.environ["https_proxy"] = "http://127.0.0.1:7890"
        # os.environ["http_proxy"] = "http://127.0.0.1:7890"

        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model.eval()  # Set model to evaluation mode
        print(f"CLIP model '{model_name}' loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Error loading CLIP model '{model_name}': {e}")
        # Consider raising the exception or returning None to handle the error upstream
        raise


# --- New CLIP Tagging Function ---
def get_clip_tags(image_path, model, processor, candidate_tags, top_n=5, threshold=0.1):
    """Generates tags for an image using CLIP based on candidate descriptions."""
    try:
        image = Image.open(image_path).convert("RGB")  # Ensure image is RGB
        # Prepare inputs using the CLIP processor
        inputs = processor(
            text=candidate_tags, images=image, return_tensors="pt", padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Convert scores to probabilities
        probs = probs.cpu().numpy()[0]  # Get probabilities for the single image

        # Pair tags with probabilities and sort
        results = sorted(zip(candidate_tags, probs), key=lambda x: x[1], reverse=True)

        # Filter by top_n and threshold
        final_tags = [
            {"tag": tag, "confidence": f"{prob*100:.2f}%"}
            for tag, prob in results[:top_n]
            if prob >= threshold
        ]
        return final_tags
    except Exception as e:
        print(f"Error processing {image_path} with CLIP: {e}")
        return []


# --- New SQLite Database Functions ---
def initialize_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Create table - storing tags as JSON text
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                type TEXT,
                tags TEXT,
                color TEXT,
                dimensions TEXT,
                last_analyzed TEXT,
                file_size INTEGER,
                thumbnail TEXT
            )
        """
        )
        # Create indices for faster querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON files (type)")
        # Note: Indexing JSON content directly might require specific SQLite versions or extensions (like FTS).
        # A simple LIKE query will work but might be slow on large datasets without FTS.
        conn.commit()
        print(f"Database initialized successfully at {DB_FILE}")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        raise  # Re-raise the exception to stop execution if DB fails
    finally:
        if conn:
            conn.close()


def add_or_update_file_record(file_data):
    """Adds or updates a file record in the SQLite database."""
    required_keys = {"path"}  # Path is essential as PRIMARY KEY
    if not required_keys.issubset(file_data.keys()):
        print(
            f"Error: Missing required key 'path' in file_data for {file_data.get('path', 'unknown')}. Skipping record."
        )
        return

    conn = None  # Initialize conn to None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # Convert tags list (list of dicts) to JSON string for storage
        tags_json = json.dumps(file_data.get("tags", []))

        cursor.execute(
            """
            INSERT OR REPLACE INTO files (path, type, tags, color, dimensions, last_analyzed, file_size, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                file_data["path"],  # Use the primary key
                file_data.get("type"),
                tags_json,  # Store JSON string
                file_data.get("color"),
                file_data.get("dimensions"),
                file_data.get("last_analyzed"),
                file_data.get("file_size"),
                file_data.get("thumbnail"),
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        print(
            f"Error adding/updating record for {file_data.get('path', 'unknown')}: {e}"
        )
    finally:
        if conn:
            conn.close()


# --- New Text Analysis using spaCy NER ---
def analyze_text_ner(text_path, nlp):
    """Extracts named entities from a text file using spaCy."""
    if not nlp:
        print("spaCy model not provided or loaded.")
        return []
    try:
        with open(text_path, "r", encoding="utf-8") as file:
            text = file.read()

        doc = nlp(text)
        # Extract entities: ent.text is the entity, ent.label_ is the type (e.g., ORG, GPE)
        tags = [
            {"tag": ent.text, "confidence": ent.label_} for ent in doc.ents
        ]  # Using confidence field for entity type

        # Optional: Add simple word frequency as fallback or supplement? (Removed for now to strictly use NER)

        # Deduplicate based on tag text and type
        unique_tags = []
        seen_tags = set()
        for tag_info in tags:
            tag_key = (
                tag_info["tag"],
                tag_info["confidence"],
            )  # Use confidence (type) for uniqueness
            if tag_key not in seen_tags:
                unique_tags.append(tag_info)
                seen_tags.add(tag_key)

        return unique_tags

    except FileNotFoundError:
        print(f"Text file not found: {text_path}")
        return []
    except Exception as e:
        print(f"Error processing text file {text_path} with spaCy NER: {e}")
        return []


def process_directory(dir_path, clip_model, clip_processor, nlp_model):
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    text_extensions = {".txt"}
    model_extensions = {".glb"}

    # Process files/folders
    for item in Path(dir_path).glob("*"):
        file_entry = None  # Initialize entry for each item

        if item.is_dir():
            print(f"Processing model folder: {item.name}")
            model_files = list(item.glob("*.glb"))
            if not model_files:
                continue
            model_file = model_files[0]
            model_rel_path = str(model_file.relative_to(dir_path))

            thumbnail_files = list(item.glob("thumbnail.*"))
            thumbnail_rel_path = (
                str(thumbnail_files[0].relative_to(dir_path))
                if thumbnail_files
                else None
            )

            tag_files = list(item.glob("*.txt"))
            model_ner_tags = []
            if tag_files:
                # Use spaCy NER for text analysis
                model_ner_tags = analyze_text_ner(tag_files[0], nlp_model)

            filename = item.name.lower()
            filename_tag = {
                "tag": filename,
                "confidence": "filename",
            }  # Use confidence for type info

            creation_time = datetime.fromtimestamp(os.path.getctime(model_file))
            year = str(creation_time.year)
            month = creation_time.strftime("%B").lower()
            year_tag = {"tag": year, "confidence": "year"}
            month_tag = {"tag": month, "confidence": "month"}
            type_tag = {"tag": "3d", "confidence": "filetype"}

            final_tags = model_ner_tags + [year_tag, month_tag, type_tag, filename_tag]

            file_entry = {
                "path": model_rel_path,  # Use relative path as key
                "type": "3d",
                "tags": final_tags,
                "last_analyzed": datetime.now().isoformat(),
                "file_size": os.path.getsize(model_file),
                "thumbnail": thumbnail_rel_path,
            }

        elif item.suffix.lower() in image_extensions:
            rel_path = str(item.relative_to(dir_path))
            print(f"Processing image: {rel_path}")
            try:
                clip_tags = get_clip_tags(
                    item, clip_model, clip_processor, CANDIDATE_IMAGE_TAGS
                )
                dominant_color = get_dominant_color(item)
                color_tag = {"tag": dominant_color, "confidence": "color"}

                filename = item.stem.lower()
                filename_tag = {"tag": filename, "confidence": "filename"}

                creation_time = datetime.fromtimestamp(os.path.getctime(item))
                year = str(creation_time.year)
                month = creation_time.strftime("%B").lower()
                year_tag = {"tag": year, "confidence": "year"}
                month_tag = {"tag": month, "confidence": "month"}
                type_tag = {"tag": "image", "confidence": "filetype"}

                with Image.open(item) as img:
                    dimensions = img.size
                dimensions_str = f"{dimensions[0]}x{dimensions[1]}"

                final_tags = clip_tags + [
                    color_tag,
                    year_tag,
                    month_tag,
                    type_tag,
                    filename_tag,
                ]

                file_entry = {
                    "path": rel_path,
                    "type": "image",
                    "tags": final_tags,
                    "color": dominant_color,
                    "dimensions": dimensions_str,
                    "last_analyzed": datetime.now().isoformat(),
                    "file_size": os.path.getsize(item),
                }
            except Exception as e:
                print(f"Error fully processing image {rel_path}: {e}")
                # Create minimal entry on error
                try:
                    with Image.open(item) as img:
                        dimensions = img.size
                    dimensions_str = f"{dimensions[0]}x{dimensions[1]}"
                    file_entry = {
                        "path": rel_path,
                        "type": "image",
                        "tags": [{"tag": "processing_error", "confidence": "error"}],
                        "dimensions": dimensions_str,
                        "last_analyzed": datetime.now().isoformat(),
                        "file_size": os.path.getsize(item),
                    }
                except Exception as inner_e:
                    print(f"Could not even get basic info for {rel_path}: {inner_e}")
                    continue  # Skip if even basic info fails

        elif item.suffix.lower() in text_extensions:
            rel_path = str(item.relative_to(dir_path))
            # Skip companion .txt files (already handled with their models)
            corresponding_folder = item.parent / item.stem
            if corresponding_folder.is_dir():
                continue
            corresponding_glb = item.with_suffix(".glb")
            if corresponding_glb.exists():
                continue

            print(f"Processing standalone text: {rel_path}")
            try:
                ner_tags = analyze_text_ner(item, nlp_model)

                filename = item.stem.lower()
                filename_tag = {"tag": filename, "confidence": "filename"}

                creation_time = datetime.fromtimestamp(os.path.getctime(item))
                year = str(creation_time.year)
                month = creation_time.strftime("%B").lower()
                year_tag = {"tag": year, "confidence": "year"}
                month_tag = {"tag": month, "confidence": "month"}
                type_tag = {"tag": "text", "confidence": "filetype"}

                final_tags = ner_tags + [year_tag, month_tag, type_tag, filename_tag]

                file_entry = {
                    "path": rel_path,
                    "type": "text",
                    "tags": final_tags,
                    "last_analyzed": datetime.now().isoformat(),
                    "file_size": os.path.getsize(item),
                }
            except Exception as e:
                print(f"Error processing text file {rel_path}: {e}")
                file_entry = {
                    "path": rel_path,
                    "type": "text",
                    "tags": [{"tag": "processing_error", "confidence": "error"}],
                    "last_analyzed": datetime.now().isoformat(),
                    "file_size": os.path.getsize(item),
                }

        elif item.suffix.lower() in model_extensions:
            # Handle standalone .glb files (not in a folder of the same name)
            corresponding_folder = item.parent / item.stem
            if corresponding_folder.is_dir():
                print(f"Skipping standalone GLB: {item.name} (handled as model folder)")
                continue  # Already handled by the is_dir() block

            rel_path = str(item.relative_to(dir_path))
            print(f"Processing standalone 3D model: {rel_path}")

            txt_path = item.with_suffix(".txt")
            model_ner_tags = []
            if txt_path.exists():
                model_ner_tags = analyze_text_ner(txt_path, nlp_model)

            filename = item.stem.lower()
            filename_tag = {"tag": filename, "confidence": "filename"}

            creation_time = datetime.fromtimestamp(os.path.getctime(item))
            year = str(creation_time.year)
            month = creation_time.strftime("%B").lower()
            year_tag = {"tag": year, "confidence": "year"}
            month_tag = {"tag": month, "confidence": "month"}
            type_tag = {"tag": "3d", "confidence": "filetype"}

            final_tags = model_ner_tags + [year_tag, month_tag, type_tag, filename_tag]

            file_entry = {
                "path": rel_path,
                "type": "3d",
                "tags": final_tags,
                "last_analyzed": datetime.now().isoformat(),
                "file_size": os.path.getsize(item),
            }

        # Add the processed entry to the database
        if file_entry:
            add_or_update_file_record(file_entry)


def main():
    # Initialize database first
    try:
        initialize_db()
    except Exception as e:
        print(f"Database initialization failed. Exiting. Error: {e}")
        return

    print("Setting up models...")
    nlp_model = None  # Initialize nlp_model
    try:
        # Load CLIP model
        clip_model, clip_processor = setup_clip_model()

        # Load spaCy NER model
        try:
            nlp_model = spacy.load("en_core_web_sm")
            print("spaCy NER model 'en_core_web_sm' loaded successfully.")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            try:
                spacy.cli.download("en_core_web_sm")
                nlp_model = spacy.load("en_core_web_sm")
                print(
                    "spaCy NER model 'en_core_web_sm' downloaded and loaded successfully."
                )
            except Exception as download_e:
                print(
                    f"Failed to download spaCy model. NER tagging will be skipped. Error: {download_e}"
                )
                # Proceed without NER if download fails, nlp_model remains None

    except Exception as e:
        print(f"Failed to load models. Exiting. Error: {e}")
        return  # Exit if models fail to load (CLIP error most likely)

    print("Processing files...")
    # Pass models to process_directory
    process_directory("data", clip_model, clip_processor, nlp_model)  # Pass nlp_model

    print(f"Done! Results saved to {DB_FILE}")  # Update final message


if __name__ == "__main__":
    # Optional: Proxy settings might be needed before model download in setup_clip_model
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    main()
