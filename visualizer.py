import streamlit as st
import sqlite3
import json
from pathlib import Path
from PIL import Image
import math

DB_FILE = "data/tags.sqlite"
DATA_DIR = Path("data")  # Base directory for relative paths in DB
ITEMS_PER_PAGE = 12


# --- Database Function --- a
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def load_data():
    """Loads data from the SQLite database and parses tags."""
    all_data = []
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT path, type, tags, color, dimensions, last_analyzed, file_size, thumbnail FROM files"
        )
        rows = cursor.fetchall()
        conn.close()

        column_names = [description[0] for description in cursor.description]

        for row in rows:
            item = dict(zip(column_names, row))
            try:
                # Parse the tags JSON string back into a list
                item["tags"] = json.loads(item["tags"]) if item["tags"] else []
            except json.JSONDecodeError:
                item["tags"] = [{"tag": "Error parsing tags", "confidence": "error"}]
            all_data.append(item)
        return all_data
    except sqlite3.Error as e:
        st.error(f"Error connecting to or reading database: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return []


# --- Main App --- a
st.set_page_config(layout="wide")
st.title("Repository Visualizer")

# Load data
data = load_data()

if not data:
    st.warning("No data found in the database or unable to load data.")
    st.stop()

# --- Sidebar Filters --- a
st.sidebar.header("Filters")

# Search term
search_term = st.sidebar.text_input("Search by Path or Tag").lower()

# File type filter
all_types = sorted(list(set(item["type"] for item in data if item.get("type"))))
selected_types = st.sidebar.multiselect(
    "File Type", options=all_types, default=all_types
)

# Tag filter
all_tags_set = set()
for item in data:
    for tag_info in item.get("tags", []):
        all_tags_set.add(tag_info.get("tag"))
all_tags_list = sorted(list(all_tags_set))
selected_tags = st.sidebar.multiselect(
    "Filter by Tags (AND logic)", options=all_tags_list, default=[]
)

# --- Filtering Logic --- a
filtered_data = data

# Filter by type
if selected_types:
    filtered_data = [
        item for item in filtered_data if item.get("type") in selected_types
    ]

# Filter by search term (path or tag text)
if search_term:
    filtered_data = [
        item
        for item in filtered_data
        if search_term in item.get("path", "").lower()
        or any(
            search_term in tag_info.get("tag", "").lower()
            for tag_info in item.get("tags", [])
        )
    ]

# Filter by selected tags (all selected tags must be present)
if selected_tags:
    filtered_data = [
        item
        for item in filtered_data
        if all(
            stag in [t.get("tag") for t in item.get("tags", [])]
            for stag in selected_tags
        )
    ]

# --- Pagination --- a
total_items = len(filtered_data)
st.write(f"Found {total_items} items.")

total_pages = math.ceil(total_items / ITEMS_PER_PAGE)
if total_pages == 0:
    total_pages = 1  # Ensure at least one page

current_page = st.number_input(
    "Page",
    min_value=1,
    max_value=total_pages,
    value=1,
    step=1,
    help=f"Showing page {1} of {total_pages}",
)

start_idx = (current_page - 1) * ITEMS_PER_PAGE
end_idx = start_idx + ITEMS_PER_PAGE
paged_data = filtered_data[start_idx:end_idx]

# --- Display Area --- a
if not paged_data:
    st.info("No items match the current filters.")
else:
    num_columns = 4  # Adjust number of columns as needed
    cols = st.columns(num_columns)
    for i, item in enumerate(paged_data):
        col = cols[i % num_columns]
        with col:
            # Determine image path
            display_path = None
            if item.get("type") == "image":
                display_path = DATA_DIR / item["path"]
            elif item.get("type") == "3d" and item.get("thumbnail"):
                display_path = DATA_DIR / item["thumbnail"]

            # Display image/thumbnail if path exists
            if display_path and display_path.exists():
                try:
                    image = Image.open(display_path)
                    st.image(image, use_container_width=True)
                except Exception as img_e:
                    st.warning(f"Could not load image: {display_path.name}\n{img_e}")
            elif (
                item.get("type") != "text"
            ):  # Show placeholder if no image/thumb unless text
                st.markdown(
                    f"<div style='height:150px; background-color:#f0f0f0; display:flex; align-items:center; justify-content:center; text-align:center; border-radius:5px; margin-bottom:5px;'>No Preview</div>",
                    unsafe_allow_html=True,
                )

            # Display Info
            st.markdown(f"**{item.get('path', 'N/A')}**")
            details = []
            if item.get("type"):
                details.append(f"Type: {item['type']}")
            if item.get("file_size"):
                details.append(f"Size: {item['file_size']:,} B")  # Format size
            if item.get("dimensions"):
                details.append(f"Dims: {item['dimensions']}")
            if item.get("color"):
                details.append(f"Color: {item['color']}")
            st.caption(" | ".join(details))

            # Display Tags
            tags_list = item.get("tags", [])
            if tags_list:
                with st.expander("Tags"):  # Use expander for potentially long tag lists
                    tag_markdown = ""
                    for tag_info in tags_list:
                        tag_name = tag_info.get("tag", "N/A")
                        tag_conf = tag_info.get("confidence", "N/A")
                        # Simple formatting: show tag and its type/confidence
                        tag_markdown += f"- **{tag_name}** ({tag_conf})\n"
                    st.markdown(tag_markdown)
            st.markdown("---")  # Separator
