"""
train.py

Trenira model za prepoznavanje rukom pisane srpske cirilice (60 klasa) na
data/combined/ (izlaz iz combine_datasets.py). Deli podatke na train/val/test
(70/15/15, posebno za svaku klasu), trenira, i na kraju cuva model i izvestaje
(grafike, matricu konfuzije, tacnost po slovu) u reports/.

Dve opcije za model (--backbone):
  - scratch: mali CNN treniran od nule (podrazumevano)
  - mobilenetv2: fine-tuning MobileNetV2 modela pretreniranog na ImageNet-u -
    bolje generalizuje na pravi rukopis jer vec "zna" opste oblike i ivice,
    ne samo ono sto nauci iz naseg (uglavnom sintetickog) dataset-a

Train/val/test podela se pamti u --split_file, tako da ako se kasnije doda
vise podataka, novi fajlovi idu samo u train skup, a val/test ostaju isti
(da ne bi neki uzorak iz test skupa naknadno zavrsio u treningu).

Za nastavak treninga postojeceg modela:
    python3 src/train.py --resume_from models/model_best.h5
"""

import argparse
import json
import os
import random

import matplotlib

matplotlib.use("Agg")  # bez potrebe za GUI-jem (radi i preko ssh/bez ekrana)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from cyrillic_alphabet import all_classes

# --------------------------------------------------------------------------- #
# Konfiguracija
# --------------------------------------------------------------------------- #

IMAGE_SIZE = 64
PRETRAINED_IMAGE_SIZE = 96  # MobileNetV2 minimalna podrzana velicina
BATCH_SIZE = 64
EPOCHS = 40
FREEZE_EPOCHS = 5  # koliko epoha se trenira samo nova glava, pre odmrzavanja backbone-a
EARLY_STOPPING_PATIENCE = 8
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_COMBINED_DIR = os.path.join(PROJECT_ROOT, "data", "combined")
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DEFAULT_SPLIT_FILE = os.path.join(DEFAULT_REPORTS_DIR, "dataset_split.json")

CLASSES = all_classes()  # [(index, letter, is_upper, class_name), ...] - fiksan redosled
CLASS_NAMES = [c[3] for c in CLASSES]  # npr. '01_lower_а' - redosled = redosled labela 0..59
LETTER_LABELS = [c[1] for c in CLASSES]  # citljivo ime za izvestaje, npr. 'а'
NUM_CLASSES = len(CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Trenira CNN za prepoznavanje srpske cirilice.")
    parser.add_argument("--combined_dir", default=DEFAULT_COMBINED_DIR)
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports_dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--split_file",
        default=None,
        help="Gde se cuva/ucitava train/val/test podela (podrazumevano <reports_dir>/dataset_split.json).",
    )
    parser.add_argument(
        "--fresh_split",
        action="store_true",
        help="Ignorisi postojeci --split_file i izracunaj potpuno nov split od nule.",
    )
    parser.add_argument(
        "--resume_from",
        default=None,
        help="Putanja do vec sacuvanog modela od kog se nastavlja trening (ukljucujuci stanje optimizatora).",
    )
    parser.add_argument(
        "--backbone",
        choices=["scratch", "mobilenetv2"],
        default="scratch",
        help="'scratch' = mali CNN od nule. 'mobilenetv2' = fine-tuning pretreniranog modela.",
    )
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=FREEZE_EPOCHS,
        help="Samo za --backbone mobilenetv2: koliko epoha treniramo sa zamrznutim backbone-om.",
    )
    return parser.parse_args()


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def _fresh_split_for_class(class_dir):
    """Racuna nov 70/15/15 split fajlova unutar jedne klase."""
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(".png"))
    paths = [os.path.join(class_dir, f) for f in files]

    train_files, temp_files = train_test_split(paths, test_size=(1.0 - TRAIN_FRACTION), random_state=SEED)
    relative_val_size = VAL_FRACTION / (VAL_FRACTION + TEST_FRACTION)
    val_files, test_files = train_test_split(temp_files, test_size=(1.0 - relative_val_size), random_state=SEED)
    return train_files, val_files, test_files


def split_dataset(combined_dir, split_file=None, fresh_split=False):
    """
    Deli fajlove po klasi na train/val/test (70/15/15), posebno za svaku klasu.

    Ako split_file vec postoji, ucitava se ta podela - stari fajlovi ostaju u
    istom skupu, a novi (npr. posle ponovnog pokretanja combine_datasets.py)
    idu samo u train, nikad u val/test, da test skup ostane "cist".

    Vraca tri liste (paths, labels): train, val, test.
    """
    previous_split = None
    if split_file and os.path.isfile(split_file) and not fresh_split:
        with open(split_file) as f:
            previous_split = json.load(f)
        print(f"Ucitavam postojecu train/val/test podelu iz {split_file} ...")

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(combined_dir, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Folder klase ne postoji: {class_dir}\nDa li si prvo pokrenula combine_datasets.py?"
            )
        current_files = set(os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(".png"))

        if previous_split is None:
            train_files, val_files, test_files = _fresh_split_for_class(class_dir)
        else:
            # zadrzi stare fajlove u istom skupu, nove dodaj samo u train
            prev_train = set(previous_split["train"].get(class_name, []))
            prev_val = set(previous_split["val"].get(class_name, []))
            prev_test = set(previous_split["test"].get(class_name, []))

            train_files = list(prev_train & current_files)
            val_files = list(prev_val & current_files)
            test_files = list(prev_test & current_files)

            already_known = prev_train | prev_val | prev_test
            new_files = sorted(current_files - already_known)
            if new_files:
                train_files.extend(new_files)

        train_paths.extend(train_files)
        train_labels.extend([label_idx] * len(train_files))
        val_paths.extend(val_files)
        val_labels.extend([label_idx] * len(val_files))
        test_paths.extend(test_files)
        test_labels.extend([label_idx] * len(test_files))

    if split_file:
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        by_class = {"train": {}, "val": {}, "test": {}}
        for name, paths in (("train", train_paths), ("val", val_paths), ("test", test_paths)):
            for path in paths:
                class_name = os.path.basename(os.path.dirname(path))
                by_class[name].setdefault(class_name, []).append(path)
        with open(split_file, "w") as f:
            json.dump(by_class, f)
        print(f"Podela sacuvana u {split_file}")

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def make_dataset(paths, labels, batch_size, shuffle, backbone="scratch"):
    """
    Cita slike sa diska i priprema ih za trening.

    Za scratch: grayscale 64x64, piksali skalirani na [0,1].
    Za mobilenetv2: slika se pretvara u RGB i skalira na 96x96 (MobileNetV2
    trazi 3 kanala i minimum tu velicinu), pa se normalizuje onako kako
    MobileNetV2 ocekuje.
    """
    is_pretrained = backbone == "mobilenetv2"
    image_size = PRETRAINED_IMAGE_SIZE if is_pretrained else IMAGE_SIZE

    def load_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_png(raw, channels=1)
        image = tf.image.resize(image, [image_size, image_size])
        if is_pretrained:
            image = tf.image.grayscale_to_rgb(image)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        else:
            image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # dekodiranje sa diska se placa samo u prvoj epohi
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 20000), seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_augmentation():
    """Blaga augmentacija (rotacija/translacija/zum). Primenjuje se samo
    tokom treninga, ne i pri testiranju."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),  # ~ +/-18 stepeni
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )


def build_model():
    """Mali CNN od nule - 3 konvoluciona bloka + BatchNorm + Dropout + L2
    regularizacija (dodato jer je bez toga model brzo overfitovao, verovatno
    zato sto sintetika iz svega 3 fonta cini vecinu dataset-a)."""
    l2 = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
    x = build_augmentation()(inputs)

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", kernel_regularizer=l2)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=l2)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model_mobilenet(image_size):
    """
    Fine-tuning MobileNetV2 pretreniranog na ImageNet-u.

    Trening ide u dve faze: prvo se backbone zamrzne i trenira se samo nova
    klasifikaciona glava, a onda se backbone odmrzne i ceo model se fino
    podesava sa mnogo nizim learning rate-om da se ne pokvari ono sto je
    MobileNetV2 vec naucio.

    Vraca (model, base_model) - base_model treba spolja da bi se posle prve
    faze postavilo base_model.trainable = True.
    """
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False  # faza 1: zamrznut

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = build_augmentation()(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def plot_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Tacnost")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("Tacnost")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Funkcija gubitka")
    axes[1].set_xlabel("Epoha")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm,
        xticklabels=LETTER_LABELS,
        yticklabels=LETTER_LABELS,
        cmap="viridis",
        square=True,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predvidjeno")
    ax.set_ylabel("Stvarno")
    ax.set_title("Matrica konfuzije - test skup")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def print_per_letter_accuracy(cm):
    print("\nTacnost po slovu (test skup), sortirano od najloseg ka najboljem:")
    per_class_acc = []
    for idx, letter in enumerate(LETTER_LABELS):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        acc = correct / total if total > 0 else float("nan")
        per_class_acc.append((letter, acc, total))

    for letter, acc, total in sorted(per_class_acc, key=lambda t: t[1]):
        print(f"  {letter:3s}  tacnost={acc:.3f}  (n={total})")


def main():
    args = parse_args()
    set_seeds()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)

    split_file = args.split_file or os.path.join(args.reports_dir, "dataset_split.json")

    print("Delim dataset na train/val/test (70/15/15, stratifikovano po klasi) ...")
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(
        args.combined_dir, split_file=split_file, fresh_split=args.fresh_split
    )
    print(f"train={len(train_paths)}  val={len(val_paths)}  test={len(test_paths)}")

    train_ds = make_dataset(train_paths, train_labels, args.batch_size, shuffle=True, backbone=args.backbone)
    val_ds = make_dataset(val_paths, val_labels, args.batch_size, shuffle=False, backbone=args.backbone)
    test_ds = make_dataset(test_paths, test_labels, args.batch_size, shuffle=False, backbone=args.backbone)

    # cuvamo kao .h5 - noviji .keras format ima bag sa ModelCheckpoint u ovoj verziji TF-a
    model_name = "model_best.h5" if args.backbone == "scratch" else "model_best_mobilenetv2.h5"
    model_path = os.path.join(args.models_dir, model_name)

    def make_callbacks():
        return [
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
        ]

    if args.resume_from:
        print(f"Ucitavam postojeci model za nastavak treninga: {args.resume_from}")
        model = tf.keras.models.load_model(args.resume_from)
        model.summary()
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=make_callbacks())
    elif args.backbone == "mobilenetv2":
        model, base_model = build_model_mobilenet(PRETRAINED_IMAGE_SIZE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        print(f"\n=== FAZA 1: trening samo nove glave ({args.freeze_epochs} epoha, backbone zamrznut) ===\n")
        history1 = model.fit(
            train_ds, validation_data=val_ds, epochs=args.freeze_epochs, callbacks=make_callbacks()
        )

        print("\n=== FAZA 2: odmrzavanje backbone-a, fino podesavanje sa niskim LR ===\n")
        base_model.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # nizi LR nego u fazi 1
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            initial_epoch=args.freeze_epochs,
            callbacks=make_callbacks(),
        )

        # spajamo istoriju obe faze u jedan grafik
        history = history1
        for key in history.history:
            history.history[key].extend(history2.history[key])
    else:
        model = build_model()
        model.summary()
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=make_callbacks())

    plot_training_curves(history, os.path.join(args.reports_dir, "training_curves.png"))

    print("\nEvaluacija na test skupu ...")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array(test_labels)

    report = classification_report(y_true, y_pred, target_names=LETTER_LABELS, digits=3)
    print("\nIzvestaj po klasama:\n")
    print(report)
    with open(os.path.join(args.reports_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")
        f.write(report)

    cm = plot_confusion_matrix(y_true, y_pred, os.path.join(args.reports_dir, "confusion_matrix.png"))
    print_per_letter_accuracy(cm)

    print(f"\nNajbolji model sacuvan u: {model_path}")
    print(f"Grafici i izvestaji sacuvani u: {args.reports_dir}")


if __name__ == "__main__":
    main()
