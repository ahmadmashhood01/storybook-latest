"""
Shared utilities for locating book template assets.
"""
import os
from pathlib import Path
from config import BOOKS_BASE_DIR

# Book path mapping - maps book names to their folder names in BOOKS_BASE_DIR
# Each book folder contains: 00.png (front), 0.png (back), 1.png (full wrap), 2-33.png (interior)
BOOK_PATHS = {
    "A True Princess": "A True Princess PNG",
    "My Animal World": "My Animal World PNG",
    "My Dinosaur World": "My Dinosaur World PNG",
    "My Little Wonder": "My Little Wonder PNG",
    "My Little Wonder - Boy": "My Little Wonder - Boy PNG",
    "Superhero": "Superhero PNG",
    "Little Farmer's Big Day": "Littler Farmer's Big Day PNG",
    "I Believe in Me": "I Believe in Me PNG",
    "My ABC Adventure": "My ABC Adventure PNG",
    "Santa's Little Helper": "Santa's Little Helper PNG",
    "The Unicorn Dreamland": "The Unicorn Dreamland PNG",
    "The World of Big Dreams": "The World of Big Dreams PNG",
}


def get_book_template_path(book_name: str, base_dir: str | None = None) -> Path:
    """
    Return the full path to the template directory for a given book name.
    """
    if book_name not in BOOK_PATHS:
        raise ValueError(f"Unknown book: {book_name}. Available books: {list(BOOK_PATHS.keys())}")

    root = base_dir or BOOKS_BASE_DIR
    book_folder_name = BOOK_PATHS[book_name]
    return Path(os.path.join(root, book_folder_name))

