"""
cyrillic_alphabet.py

Azbuka i imena foldera na jednom mestu, da ih generate_synthetic.py,
combine_datasets.py i train.py koriste isto.
"""

import os
import re

# 30 malih slova srpske cirilice, u standardnom azbucnom redosledu
LOWER_LETTERS = list("абвгдђежзијклљмнњопрстћуфхцчџш")
# Odgovarajuca velika slova (Python .upper() ispravno mapira i Ђ Ј Љ Њ Ћ Џ)
UPPER_LETTERS = [ch.upper() for ch in LOWER_LETTERS]

NUM_CLASSES = len(LOWER_LETTERS) * 2  # 60

_CLASS_NAME_RE = re.compile(r"^(\d{2})_(lower|upper)_(.)$")


def build_class_name(index, letter, is_upper):
    """
    Vraca ime foldera za dato slovo, npr. '01_lower_а' / '01_upper_А'.

    Mora imati ovaj prefiks jer Mac (podrazumevano) ne pravi razliku izmedju
    velikih i malih slova u imenima foldera, pa bi 'А' i 'а' bez prefiksa
    zavrsili u istom folderu i pomesali se.
    """
    case_tag = "upper" if is_upper else "lower"
    return f"{index:02d}_{case_tag}_{letter}"


def parse_class_name(class_name):
    """Obrnuto od build_class_name - iz imena foldera vraca (index, is_upper, letter)."""
    m = _CLASS_NAME_RE.match(class_name)
    if not m:
        raise ValueError(f"Neispravno ime klase: {class_name!r}")
    index, case_tag, letter = m.groups()
    return int(index), case_tag == "upper", letter


def all_classes():
    """Vraca svih 60 klasa u fiksnom redosledu (za svako slovo prvo malo pa
    veliko), kao listu (index, letter, is_upper, class_name)."""
    classes = []
    for index, letter in enumerate(LOWER_LETTERS, start=1):
        for letter_variant, is_upper in ((letter, False), (letter.upper(), True)):
            classes.append((index, letter_variant, is_upper, build_class_name(index, letter_variant, is_upper)))
    return classes


def assert_no_case_collision(root_dir, expected_count=NUM_CLASSES):
    """
    Proverava da root_dir stvarno ima 60 odvojenih foldera. Ako ih ima manje,
    znaci da su se veliko i malo slovo (npr. 'А' i 'а') spojili u jedan folder
    jer fajl-sistem ne pravi razliku medju njima - tada dataset treba ponovo
    izvuci na case-sensitive particiji, npr.:
        hdiutil create -size 500m -fs "Case-sensitive APFS" -volname CirilicaCS ~/cirilica_cs.dmg
        hdiutil attach ~/cirilica_cs.dmg
        ditto -xk dataset.zip /Volumes/CirilicaCS/
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Folder ne postoji: {root_dir}")

    entries = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if len(entries) < expected_count:
        raise RuntimeError(
            f"Ocekivano {expected_count} foldera u {root_dir}, pronadjeno samo {len(entries)}.\n"
            "Verovatno su se veliko i malo slovo pomesali u isti folder - izvuci\n"
            "dataset ponovo na case-sensitive particiji (vidi komentar iznad)."
        )
