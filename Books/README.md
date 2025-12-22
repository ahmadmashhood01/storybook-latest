# Books Template Directory

This directory contains the book template folders used by the personalized storybook generator.

## Structure

Each book folder contains:
- `00.png` - Front cover only (for AI personalization)
- `0.png` - Back cover only (unchanged)
- `1.png` - Full wrap cover (front + spine + back) - used for spine extraction
- `2.png` through `33.png` - Interior pages
- `[BookName].pdf` - Complete book PDF (if available)

## Books

The following 12 book templates are available:
1. A True Princess PNG
2. I Believe in Me PNG
3. Littler Farmer's Big Day PNG
4. My ABC Adventure PNG
5. My Animal World PNG
6. My Dinosaur World PNG
7. My Little Wonder - Boy PNG
8. My Little Wonder PNG
9. Santa's Little Helper PNG
10. Superhero PNG
11. The Unicorn Dreamland PNG
12. The World of Big Dreams PNG

## Notes

- All PNG and PDF files are tracked via Git LFS due to their large size
- The `BOOKS_BASE_DIR` environment variable can be used to override the default path
- See `config.py` for configuration details

