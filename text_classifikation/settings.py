from os.path import dirname, join, expanduser, isdir
from os import makedirs
#import spacy

#nlp = spacy.load('en')

# https://drive.google.com/uc?id=1saFGKezSFgH-5YjsQX_yiWes41xqbHrf&export=download

MODELS_PATH = join(expanduser("~"), "text_classifikation","models")
DATA_PATH = join(expanduser("~"), "text_classifikation", "data")

if not isdir(MODELS_PATH):
    makedirs(MODELS_PATH)

if not isdir(DATA_PATH):
    makedirs(DATA_PATH)

