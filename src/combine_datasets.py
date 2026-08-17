"""
combine_datasets.py

Spaja realni dataset (skenirane forme, isecene celije, jedan folder po slovu)
sa sintetickim uzorcima iz generate_synthetic.py u jedan dataset za treniranje
(data/combined/).

--root_dir mora biti izvucen na case-sensitive particiji (vidi
cyrillic_alphabet.assert_no_case_collision) da se veliko i malo slovo ne bi
pomesali u isti folder.

Realni fajlovi se kopiraju (uz deduplikaciju po sadrzaju - Google Drive ume
da napravi 'ime(1).png' kopije istog fajla). Sinteticki fajlovi se hardlinkuju
umesto kopiraju, jer ih ima na desetine hiljada pa bi kopiranje trosilo puno
prostora na disku bez potrebe.

Sve slike u izlaznom folderu dobijaju prefiks 'real__' ili 'synth__' da se
kasnije zna odakle su.
"""

import argparse
import hashlib
import os
import shutil

from cyrillic_alphabet import all_classes, assert_no_case_collision

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, "data", "synthetic")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "combined")


def parse_args():
    parser = argparse.ArgumentParser(description="Kombinuje realni i sinteticki dataset cirilice.")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Putanja do realnog dataset-a (folder po slovu, npr. .../dataset_cirilica). "
        "Mora biti izvucen na case-sensitive particiji.",
    )
    parser.add_argument("--synthetic_dir", default=DEFAULT_SYNTHETIC_DIR, help="Izlaz generate_synthetic.py.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Gde se cuva kombinovan dataset.")
    parser.add_argument(
        "--max_synthetic_per_class",
        type=int,
        default=None,
        help="Opciono ogranicenje broja sintetickih uzoraka po klasi (podrazumevano: svi dostupni).",
    )
    return parser.parse_args()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def list_image_files(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(IMAGE_EXTENSIONS))


def copy_real_samples(real_dir, dest_dir):
    """Kopira realne uzorke, preskace fajlove koji su bas isti (isti sadrzaj).
    Vraca (broj_sacuvanih, broj_odbacenih_duplikata)."""
    files = list_image_files(real_dir)
    seen_hashes = set()
    saved, duplicates = 0, 0

    for fname in files:
        src_path = os.path.join(real_dir, fname)
        h = file_hash(src_path)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)

        dest_path = os.path.join(dest_dir, f"real__{fname}")
        shutil.copy2(src_path, dest_path)
        saved += 1

    return saved, duplicates


def link_synthetic_samples(synth_dir, dest_dir, max_count=None):
    """Hardlinkuje sinteticke uzorke (ili kopira ako hardlink ne uspe).
    Vraca broj fajlova."""
    files = list_image_files(synth_dir)
    if max_count is not None:
        files = files[:max_count]

    saved = 0
    for fname in files:
        src_path = os.path.join(synth_dir, fname)
        dest_path = os.path.join(dest_dir, f"synth__{fname}")
        try:
            os.link(src_path, dest_path)
        except OSError:
            shutil.copy2(src_path, dest_path)
        saved += 1

    return saved


def main():
    args = parse_args()

    assert_no_case_collision(args.root_dir)

    if os.path.isdir(args.output_dir):
        print(f"Brisem postojeci {args.output_dir} pre ponovnog kombinovanja ...")
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    stats = []  # (class_name, letter, real_saved, real_dupes, synth_saved)

    for index, letter, is_upper, class_name in all_classes():
        real_dir = os.path.join(args.root_dir, letter)
        synth_dir = os.path.join(args.synthetic_dir, class_name)
        dest_dir = os.path.join(args.output_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        if not os.path.isdir(real_dir):
            print(f"[UPOZORENJE] Nema realnog foldera za '{letter}' ({real_dir}), preskacem realne uzorke.")
            real_saved, real_dupes = 0, 0
        else:
            real_saved, real_dupes = copy_real_samples(real_dir, dest_dir)

        if not os.path.isdir(synth_dir):
            print(f"[UPOZORENJE] Nema sintetickog foldera za '{letter}' ({synth_dir}), preskacem sinteticke uzorke.")
            synth_saved = 0
        else:
            synth_saved = link_synthetic_samples(synth_dir, dest_dir, args.max_synthetic_per_class)

        print(
            f"{class_name:20s} ({letter})  real={real_saved:5d} (dupli odbaceni={real_dupes:3d})  "
            f"sinteticki={synth_saved:5d}"
        )
        stats.append((class_name, letter, real_saved, real_dupes, synth_saved))

    print("\n" + "=" * 70)
    print("STATISTIKA KOMBINOVANJA")
    print("=" * 70)
    total_real = sum(s[2] for s in stats)
    total_dupes = sum(s[3] for s in stats)
    total_synth = sum(s[4] for s in stats)
    total_all = total_real + total_synth

    print(f"{'Klasa':20s} {'Slovo':6s} {'Realno':8s} {'Duplikati':10s} {'Sinteticki':12s} {'Ukupno':8s}")
    for class_name, letter, real_saved, real_dupes, synth_saved in stats:
        print(
            f"{class_name:20s} {letter:6s} {real_saved:<8d} {real_dupes:<10d} "
            f"{synth_saved:<12d} {real_saved + synth_saved:<8d}"
        )

    print("-" * 70)
    print(f"Ukupno klasa: {len(stats)}")
    print(f"Ukupno realnih uzoraka: {total_real} (odbaceno duplikata: {total_dupes})")
    print(f"Ukupno sintetickih uzoraka: {total_synth}")
    print(f"Ukupno u kombinovanom dataset-u: {total_all}")
    print(f"Prosecno po klasi: {total_all / len(stats):.1f}")


if __name__ == "__main__":
    main()
