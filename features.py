import nltk
import re
from nltk.stem import WordNetLemmatizer

# Feature engineering
punctuations = [".", "," , "-", "?", "!", "/" , "*", "~", ">" , "%", ":", "*", "(", ")", "[", "]", "{", "}", ">", "<", "=", "+", "_", "|", "\\", "^", "&", "$", "#", "@", "`", "'", '"', ";"]
quantity = ["ml" , "l" , "mg", "g", "kg", "cm", "mm", "m", "xg" , "ng" , "in", "nm" , "ul" , "rpm", "nmol" , "c" , "mls" ]

def lowercase(t):
    return t.lower()

def has_punctuation(t,punctuations = punctuations):
    # check if punctuation is in the word
    for p in punctuations:
        if p in t:
            return True
    return False

def isFirstUpper(t):
    return t[0].isupper()

def isAllUpper(t):
    return t.isupper()

def number(t):
    return t.isdigit()

def lemma(t):
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(t)

def get_pos_tag(t):
    # get pos tag
    return nltk.pos_tag([t])[0][1]

def is_quantity(t, quantity = quantity):
    # check if word is a quantity preceded by a number
    # lower
    t = t.lower()
    if("." in t):
        # roundoff to 0 decimal places
        t = re.sub(r'\d+\.\d+', lambda x: str(int(round(float(x.group()), 0))), t)
    # remove all non-alphanumeric characters
    if(t.isalnum() == False):
        # remove commas
        t = re.sub(r'\W+', '', t)
    # check if quantity
    # regex pattern 
    regex_pattern = r"^\d+(" + "|".join(quantity) + r")$"

    return re.match(regex_pattern, t) is not None

def has_multiple_words(t):
    # check if word has multiple words
    if ("/" in t):
        return True
    elif ("-" in t):
        return True
    return False

def remove_numbers(t):
    # check if t has a digit
    if (any(char.isdigit() for char in t)):
        t = re.sub(r'\d+', "0", t)
        return t

    return t

def is_chemical(t):
    # check if word is a chemical
    regex_pattern = r"^[A-Z][a-z]?\d*"
    return re.match(regex_pattern, t) is not None

def is_device(t):

    chemical_Devices = {'pipette': 0,
    'tube': 0,
    'beaker': 0,
    'flask': 0,
    'cylinder': 0,
    'burette': 0,
    'funnel': 0,
    'clamp': 0,
    'brush': 0,
    'holder': 0,
    'stand': 0,
    'burner': 0,
    'dish': 0,
    'dropper': 0,
    'tong': 0,
    'plate': 0,
    'spatula': 0,
    'condenser': 0,
    'filter': 0,
    'mortar': 0,
    'pestle': 0,
    'pH': 0,
    'colorimeter': 0,
    'potentiometer': 0,
    'machine': 0,
    'vivaspin20': 0,
    'kit': 0,
    'spectrometer': 0,
    'microcentrifuge': 0,
    'paper': 0,
    'tubes': 0,
    'filters': 0,
    'chamber': 0,
    'dialyzer': 0,
    'syringe': 0,
    'paintbrush': 0,
    'beckman': 0,
    'apparatus': 0,
    'diffuser': 0,
    'pump': 0,
    'piptte': 0,
    'magnet': 0,
    'thermocycler': 0,
    'centrifuge': 0,
    'strainer': 0,
    'cytometer': 0,
    'rotors': 0,
    'microscope': 0,
    'plunger': 0,
    'needle': 0,
    'forceps': 0,
    'sonicator': 0,
    'homogenizer': 0}

    if t in chemical_Devices:
        return True
    return False

def isMixedcaps(t):
    # check if word is mixed caps
    return (t != t.lower())

def remove_punctuations(text_content):

    try:
        # remove punctuation
        text_content = re.sub(r'[^\w\s]'," ",text_content)
        text_content = re.sub(r'[_]'," ",text_content)
    except:
        print(f"Error in removing punctuation : {text_content}")
        return text_content

    return text_content

def preprocess(t):
    # preprocess the word
    # lower
    t = remove_numbers(t)
    # lemmatize
    # t = lemma(t)
    
    return t