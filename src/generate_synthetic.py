"""
generate_synthetic.py

Generise sinteticke uzorke rukom pisane cirilice da bismo prosirili mali
realni dataset. TRDG ima opciju is_handwritten=True, ali ta opcija radi samo
za latinicu, pa umesto toga koristimo obicne fontove koji lice na rukopis
(Neucha, Underdog, Pangolin - proverila sam da sadrze sva slova, ukljucujuci
Đ Ј Љ Њ Ћ Џ). Da bi slike vise licile na pravi rukopis, dodajemo nasumicnu
velicinu, blagu rotaciju, zamucenje i malo suma na kraju.

Napomena: TRDG 1.8.0 ne radi sa Pillow 10+, treba Pillow < 10:
    pip install "pillow==9.5.0"
"""

import os
import random

import numpy as np
from PIL import Image, ImageOps
from trdg.generators import GeneratorFromStrings

from cyrillic_alphabet import LOWER_LETTERS, UPPER_LETTERS, build_class_name

# --------------------------------------------------------------------------- #
# Konfiguracija
# --------------------------------------------------------------------------- #

SAMPLES_PER_LETTER = 5000  # minimum uzoraka po slovu
IMAGE_SIZE = 64  # finalna velicina slike (kvadratna, grayscale)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FONTS_DIR = os.path.join(PROJECT_ROOT, "fonts")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")

FONT_FILES = [
    "Neucha-Regular.ttf",
    "Underdog-Regular.ttf",
    "Pangolin-Regular.ttf",
]
FONT_PATHS = [os.path.join(FONTS_DIR, name) for name in FONT_FILES]

# Nekoliko tamnih nijansi da simuliramo razlicite hemijske olovke/penkala
INK_COLORS = ["#000000", "#0b0b0b", "#1a1a2e", "#101820"]

RANDOM_SEED = 42  # radi reproduktivnosti rezultata u radu


def build_generator(letter, font_path):
    """Pravi TRDG generator za jedan uzorak slova, sa nasumicnom velicinom/
    rotacijom/zamucenjem da svaki uzorak izgleda malo drugacije."""
    return GeneratorFromStrings(
        strings=[letter],
        count=1,
        fonts=[font_path],
        size=random.randint(80, 128),
        skewing_angle=12,
        random_skew=True,
        blur=random.randint(0, 1),
        random_blur=True,
        background_type=1,  # cista bela pozadina
        margins=(6, 6, 6, 6),
        fit=True,
        text_color=random.choice(INK_COLORS),
    )


def add_gaussian_noise(image_array, sigma=6.0):
    """Dodaje malo suma na sliku da ne bude previse "cista"."""
    noise = np.random.normal(loc=0.0, scale=sigma, size=image_array.shape)
    noisy = image_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def to_square_canvas(image, size):
    """Prebacuje sliku u grayscale i centrira je na kvadratno belo platno,
    bez razvlacenja slova."""
    gray = image.convert("L")
    gray = ImageOps.pad(gray, (size, size), color=255, centering=(0.5, 0.5))
    return gray


def generate_letter_samples(letter, class_dir, num_samples):
    """Generise num_samples uzoraka za jedno slovo i cuva ih kao PNG u class_dir."""
    os.makedirs(class_dir, exist_ok=True)
    saved = 0

    for i in range(num_samples):
        font_path = random.choice(FONT_PATHS)
        generator = build_generator(letter, font_path)
        image, _ = next(generator)

        canvas = to_square_canvas(image, IMAGE_SIZE)
        noisy_array = add_gaussian_noise(np.array(canvas))
        final_image = Image.fromarray(noisy_array, mode="L")

        out_path = os.path.join(class_dir, f"{i:04d}.png")
        final_image.save(out_path)
        saved += 1

    return saved


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for font_path in FONT_PATHS:
        if not os.path.isfile(font_path):
            raise FileNotFoundError(
                f"Font nije pronadjen: {font_path}\n"
                f"Proveri da li se fajl nalazi u {FONTS_DIR}"
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats = []  # (folder_name, letter, broj_sacuvanih_uzoraka)

    all_letters = list(enumerate(LOWER_LETTERS, start=1))
    for index, letter in all_letters:
        for letter_variant, is_upper in ((letter, False), (letter.upper(), True)):
            class_name = build_class_name(index, letter_variant, is_upper)
            class_dir = os.path.join(OUTPUT_DIR, class_name)

            print(f"Generisanje uzoraka za '{letter_variant}' -> {class_name} ...")
            saved = generate_letter_samples(letter_variant, class_dir, SAMPLES_PER_LETTER)
            stats.append((class_name, letter_variant, saved))

    print("\n" + "=" * 50)
    print("STATISTIKA GENERISANJA")
    print("=" * 50)
    total = 0
    for class_name, letter_variant, saved in stats:
        print(f"{class_name:20s} ({letter_variant})  {saved} uzoraka")
        total += saved
    print("-" * 50)
    print(f"Ukupno klasa: {len(stats)}")
    print(f"Ukupno uzoraka: {total}")
    print(f"Prosecno po klasi: {total / len(stats):.1f}")


if __name__ == "__main__":
    main()
