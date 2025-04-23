import random, re, string
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def get_synonyms(word):
    """Return a list of synonyms for a word from WordNet (excluding itself)."""
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ')
            if name.lower() != word.lower():
                syns.add(name)
    return list(syns)

def synonym_replacement_rate(text, rate=0.01):
    """
    Replace approximately `rate` fraction of non-stopwords in the text with synonyms.

    Args:
        text (str): Input text (sentence or larger).
        rate (float): Fraction of replaceable words to swap out (default 0.01 = 1%).

    Returns:
        str: Augmented text.
    """
    words = word_tokenize(text)
    candidates = [i for i, w in enumerate(words)
                  if w.isalpha() and w.lower() not in stop_words]
    
    n_replace = int(len(candidates) * rate)
    if n_replace < 1:
        return text  
    random.shuffle(candidates)
    replaced = 0
    for idx in candidates:
        syns = get_synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
            replaced += 1
        if replaced >= n_replace:
            break

    return ' '.join(words)

def aug_html_entities(text: str, p=0.1) -> str:
    entities = {"'": "&#39;", '"': "&quot;", "&": "&amp;"}
    for ch, ent in entities.items():
        if random.random() < p:
            text = text.replace(ch, ent)
    return text

def aug_html_entities(text: str, p=0.1) -> str:
    entities = {"'": "&#39;", '"': "&quot;", "&": "&amp;"}
    for ch, ent in entities.items():
        if random.random() < p:
            text = text.replace(ch, ent)
    return text

def aug_word_dup(text: str, p=0.05) -> str:
    words = text.split()
    if words and random.random() < p:
        i = random.randrange(len(words))
        words.insert(i, words[i])
    return " ".join(words)

def aug_case_swap(text: str, p=0.1) -> str:
    return "".join(c.upper() if random.random() < p else c.lower() for c in text)

def aug_punct_space(text: str, p=0.05) -> str:
    out = []
    for c in text:
        if c.isalnum() and random.random() < p:
            out.append(c + random.choice(string.punctuation))
        else:
            out.append(c)
    s = "".join(out)
    return re.sub(r" ", lambda m: " " + (" " if random.random() < p else ""), s)

def aug_truncate(text: str, p=0.1) -> str:
    if random.random() < p and len(text) > 20:
        cut = int(len(text) * random.uniform(0.7, 0.9))
        return text[:cut]
    return text

def aug_char_swap(text: str, p=0.02) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if random.random() < p:
            chars[i], chars[i+1] = chars[i+1], chars[i]
    return "".join(chars)

def augment_text(text: str) -> str:
    aug_funcs = [
        aug_html_entities,
        aug_word_dup,
        aug_case_swap,
        aug_punct_space,
        aug_truncate,
        aug_char_swap,
        synonym_replacement_rate
    ]

    n = random.randint(1, 5)
    chosen = random.sample(aug_funcs, k=n)

    for fn in chosen:
        text = fn(text)
    return text