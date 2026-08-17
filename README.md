# cirilica-ocr

OCR za prepoznavanje rukom pisane srpske ćirilice (blok štampana slova) — diplomski rad, FTN Novi Sad, Primenjene računarske nauke i informatika.

## Šta radi

Prepoznaje pojedinačna rukom pisana slova srpske ćirilice (30 malih + 30 velikih = 60 klasa) isečena sa skeniranih formi. Mali skup realnih uzoraka je kombinovan sa sintetički generisanim rukopisom (TRDG), a model je dobijen fine-tuning-om pretreniranog MobileNetV2.

## Napomena o platformi

Projekat je rađen i testiran na macOS-u. Trebalo bi da radi i na Windows-u, uz jednu bitnu razliku: i macOS i Windows (NTFS) po defaultu ne prave razliku između velikih i malih slova u imenima foldera. Ako se sirovi dataset (folder po slovu: а, А, б, Б...) ikad ponovo raspakuje, to mora da se radi na fajl-sistemu koji tu razliku pravi (case-sensitive) — na Mac-u je za to korišćena privremena disk image particija (`hdiutil create -fs "Case-sensitive APFS"`), na Windows-u bi trebalo koristiti WSL ili `fsutil.exe setCaseSensitiveInfo`.

## Struktura projekta

```
cirilica-ocr/
├── src/
│   ├── cyrillic_alphabet.py   # azbuka i imena foldera, deljeno izmedju skripti
│   ├── generate_synthetic.py  # generise sinteticke uzorke rukopisa
│   ├── combine_datasets.py    # spaja realne i sinteticke uzorke
│   └── train.py                # trening modela
├── fonts/                      # rukopisni fontovi za sintetiku (+ licence)
├── data/
│   ├── synthetic/               # generise generate_synthetic.py (nije u git-u)
│   └── combined/                 # generise combine_datasets.py (nije u git-u)
├── models/                      # sacuvani istrenirani modeli
└── reports/                     # grafici, matrica konfuzije, izvestaji
```

## Instalacija

Python 3.8, virtuelno okruzenje:

```bash
python3 -m venv diplomski_env
source diplomski_env/bin/activate       # na Windows-u: diplomski_env\Scripts\activate
pip install trdg tensorflow opencv-python "pillow==9.5.0" numpy matplotlib seaborn scikit-learn scipy gdown
```

Mora biti Pillow < 10 (npr. 9.5.0) — novije verzije nisu kompatibilne sa TRDG 1.8.0.

## Kako se koristi

**1. Generisanje sintetičkih uzoraka**

```bash
python3 src/generate_synthetic.py
```

Podrazumevano generise 5000 uzoraka po slovu u `data/synthetic/`.

**2. Spajanje sa realnim podacima**

```bash
python3 src/combine_datasets.py --root_dir /putanja/do/dataset_cirilica --max_synthetic_per_class 1000
```

`--root_dir` mora biti folder sa 60 podfoldera (po jedan za svako slovo), izvučen na case-sensitive fajl-sistemu (vidi napomenu o platformi).

**3. Trening**

```bash
# mali CNN od nule
python3 src/train.py

# fine-tuning pretreniranog MobileNetV2 (bolji rezultati)
python3 src/train.py --backbone mobilenetv2
```

Za nastavak treninga postojećeg modela: `python3 src/train.py --resume_from models/model_best.h5`

## Rezultati

| Model | Test accuracy |
|---|---|
| Originalni CNN, samo realni podaci | 53% |
| CNN od nule + sintetika | 73.6% |
| MobileNetV2 fine-tuning | **95.1%** |

Detaljni izveštaji (matrica konfuzije, tačnost po slovu) su u `reports/`.

## Fontovi

Neucha, Underdog i Pangolin — sa Google Fonts, licencirani pod SIL Open Font License 1.1 (vidi `fonts/OFL_LICENSES.txt`).
