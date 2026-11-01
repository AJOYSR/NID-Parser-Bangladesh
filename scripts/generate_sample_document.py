#!/usr/bin/env python3
"""Generate a synthetic document image for testing the parser.

Contributors must never share a real identity document, so this script draws a
fake one: the layout and field labels of a Bangladeshi NID, BO account
statement, or TIN certificate, filled with obviously invented values.

Use the output to try the API, to reproduce a bug, or to attach to an issue.

Usage::

    python scripts/generate_sample_document.py --type NID --output sample_nid.png
    python scripts/generate_sample_document.py --type BO --output sample_bo.png
    python scripts/generate_sample_document.py --type TIN --name "JANE ROE"

Then send it to a running server::

    curl -X POST http://localhost:8000/extract-nid-info/ \
         -F "type=NID" -F "file=@sample_nid.png"
"""

import argparse

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1000, 640
BACKGROUND = (248, 248, 244)
INK = (20, 20, 20)
ACCENT = (26, 82, 46)

# Fonts that ship with common systems; we fall back to PIL's bitmap default.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
)


def load_font(size: int) -> ImageFont.ImageFont:
    """Return the first available TrueType font at the requested size."""
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_nid(draw: ImageDraw.ImageDraw, name: str, dob: str, number: str) -> None:
    """Draw a National ID card layout."""
    draw.rectangle([(0, 0), (WIDTH, 92)], fill=ACCENT)
    draw.text((30, 20), "Government of the People's Republic of Bangladesh", font=load_font(26), fill=(255, 255, 255))
    draw.text((30, 56), "NATIONAL ID CARD", font=load_font(24), fill=(255, 255, 255))

    # Photo placeholder.
    draw.rectangle([(40, 140), (250, 420)], outline=INK, width=3)
    draw.text((95, 265), "PHOTO", font=load_font(28), fill=(140, 140, 140))

    label = load_font(26)
    value = load_font(30)

    draw.text((300, 150), "Name:", font=label, fill=INK)
    draw.text((300, 185), name, font=value, fill=INK)

    draw.text((300, 250), "Date of Birth", font=label, fill=INK)
    draw.text((300, 285), dob, font=value, fill=INK)

    draw.text((300, 350), "ID NO:", font=label, fill=INK)
    draw.text((300, 385), number, font=value, fill=INK)

    draw.line([(40, 480), (WIDTH - 40, 480)], fill=INK, width=2)
    draw.text((40, 510), "SYNTHETIC TEST DOCUMENT - NOT A REAL ID", font=load_font(24), fill=(190, 40, 40))


def draw_bo(draw: ImageDraw.ImageDraw, name: str, number: str) -> None:
    """Draw a BO (Beneficiary Owner) account statement header."""
    draw.text((50, 60), "DEPOSITORY PARTICIPANT STATEMENT", font=load_font(34), fill=INK)
    draw.line([(50, 115), (WIDTH - 50, 115)], fill=ACCENT, width=4)

    label = load_font(26)
    value = load_font(30)

    draw.text((50, 180), "Account Holder:", font=label, fill=INK)
    draw.text((50, 215), name, font=value, fill=INK)

    draw.text((50, 300), "BO Account Number:", font=label, fill=INK)
    draw.text((50, 338), number, font=value, fill=INK)

    draw.text((50, 430), "Statement Period: 01/01/2025 to 31/03/2025", font=label, fill=INK)
    draw.text((50, 540), "SYNTHETIC TEST DOCUMENT - NOT A REAL STATEMENT", font=load_font(24), fill=(190, 40, 40))


def draw_tin(draw: ImageDraw.ImageDraw, name: str, number: str) -> None:
    """Draw a TIN certificate layout."""
    draw.text((50, 50), "NATIONAL BOARD OF REVENUE", font=load_font(32), fill=INK)
    draw.text((50, 95), "TAXPAYER'S IDENTIFICATION NUMBER CERTIFICATE", font=load_font(26), fill=ACCENT)
    draw.line([(50, 140), (WIDTH - 50, 140)], fill=ACCENT, width=4)

    label = load_font(26)
    value = load_font(34)

    draw.text((50, 210), "Name of the Taxpayer:", font=label, fill=INK)
    draw.text((50, 248), name, font=value, fill=INK)

    draw.text((50, 340), "TIN:", font=label, fill=INK)
    draw.text((50, 378), number, font=value, fill=INK)

    draw.text((50, 480), "Date of Issue: 15/03/2025", font=label, fill=INK)
    draw.text((50, 560), "SYNTHETIC TEST DOCUMENT - NOT A REAL CERTIFICATE", font=load_font(24), fill=(190, 40, 40))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", choices=("NID", "BO", "TIN"), default="NID", help="Document type to draw")
    parser.add_argument("--output", help="Output path (default: sample_<type>.png)")
    parser.add_argument("--name", default="MD: JOHN DOE", help="Name to print on the document")
    parser.add_argument("--dob", default="15/03/1990", help="Date of birth (NID only)")
    parser.add_argument("--number", help="Document number (defaults to an obviously fake value)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    defaults = {"NID": "1234567890123", "BO": "1234 5678 9012 3456", "TIN": "123456789012"}
    number = args.number or defaults[args.type]
    output = args.output or f"sample_{args.type.lower()}.png"

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    if args.type == "NID":
        draw_nid(draw, args.name, args.dob, number)
    elif args.type == "BO":
        draw_bo(draw, args.name, number)
    else:
        draw_tin(draw, args.name, number)

    image.save(output)
    print(f"Wrote {output} ({args.type})")
    print("This is a synthetic document. Do not replace its values with real ones before sharing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
