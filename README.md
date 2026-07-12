# Task 1: Language Translation Tool

A console-based translation tool built for the CodeAlpha Artificial
Intelligence Internship.

**Developed by:** Fahad Ahmad
**Student ID:** CA/DF1/189817

## What it does

- Displays a menu of supported languages.
- Lets the user pick a source and target language.
- Translates entered text using the Google Translate engine (via the
  `deep-translator` library — no API key required).
- Optionally reads the translation aloud (offline text-to-speech).
- Optionally copies the translation to the clipboard.
- Loops so the user can translate multiple pieces of text in one session.

## Requirements

- Python 3.8+
- **Internet connection** (required — translation is done through an online service)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## How to run

```bash
python translator.py
```

## Example session

```
Supported Languages:
------------------------------
  1. English (en)
  2. Urdu (ur)
  3. Spanish (es)
  ...

Choose SOURCE language number: 1
Choose TARGET language number: 2

Enter the text you want to translate: Good morning, how are you?

Translated Text:
------------------------------
صبح بخیر، آپ کیسے ہیں؟
------------------------------

Play the translation as speech? (y/n): y
Copy the translation to clipboard? (y/n): y
  Copied to clipboard!

Translate something else? (y/n): n
```

## Adding more languages

Add entries to the `LANGUAGES` dictionary at the top of `translator.py`
using valid Google Translate language codes.

## Project structure

```
Task1_Language_Translation_Tool/
├── translator.py       # main script
├── requirements.txt
├── README.md
└── .gitignore
```

## Key concepts demonstrated

- Using a translation API/library (`deep-translator`)
- Building a simple, menu-driven console UI
- Optional feature integration (text-to-speech, clipboard)
- Error handling for network/API failures
