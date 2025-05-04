# Image and Text Search Engine

A search engine that analyzes images and text files, extracting features like dominant colors, image classifications, and text topics. Results are stored in a searchable SQLite database with an interactive web interface.

## Features

### Image Analysis
- **CLIP-based Image Classification**: Using OpenAI's CLIP model for flexible zero-shot learning
  - No fixed category limitations
  - Customizable candidate tags for architectural content
  - Confidence scores for each tag
- Dominant color extraction using PIL
- Image dimensions and metadata

### Text Analysis
- **Named Entity Recognition (NER)** using spaCy
  - Semantic understanding of text content
  - Entity type classification (ORG, GPE, PERSON, etc.)
  - Structured tag extraction with context
- Basic tokenization
- Keyword extraction
- Word frequency analysis

### Data Storage
- **SQLite Database** (`data/tags.sqlite`)
  - Efficient querying and indexing
  - Structured data storage
  - Support for complex queries
  - Better performance with large datasets

### Web Interface
- **Interactive Visualization** using Streamlit
  - Real-time search and filtering
  - Visual results display
  - Color-coded tags
  - File previews
  - Sidebar filters for:
    - Text search
    - File type selection
    - Tag-based filtering
  - Grid layout with pagination
  - Detailed file information display

## File Structure

```
.
├── data/            # Directory for files to analyze
│   └── tags.sqlite  # Generated analysis results in SQLite format
├── visualizer.py    # Interactive web interface using Streamlit
├── tag.py          # Analysis engine
├── imagenet_classes.txt
└── requirements.txt # Python dependencies
```

## How to Run the Project

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Run the Analysis Script:**
   ```bash
   python tag.py
   ```
   This will analyze files in the `data/` directory and store results in `data/tags.sqlite`.

3. **Start the Visualization Interface:**
   ```bash
   streamlit run visualizer.py
   ```
   This will launch a web interface at `http://localhost:8501`.

## Notes

- The script processes `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.txt`, and `.glb` files
- Initial model downloads (CLIP and spaCy) may take time and require stable internet connection
- For PyTorch installation issues, visit [PyTorch website](https://pytorch.org/) for system-specific instructions





















