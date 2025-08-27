from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from .datatypes import Document, Sentence

_WORD_RE = re.compile(r"""[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?""")  # simple token rule

STOPWORDS = {
    # minimal English stopword set (extend as needed)
    'the','a','an','and','or','but','if','then','else','for','to','of','in','on','at','by','with','as',
    'is','are','was','were','be','been','being','this','that','these','those','it','its','from','into',
    'we','you','they','he','she','i','me','my','your','our','their','his','her','them','us','do','does',
    'did','not','no','so','than','too','very','can','could','should','would','will','shall'
}

@dataclass
class PreprocessConfig:
    lowercase: bool = True
    remove_stopwords: bool = True
    stemming: bool = True
    title_as_first_sentence: bool = True  # use S1 as pseudo-title if none provided

def _simple_stem(token: str) -> str:
    # Very light stemmer for English as a placeholder; avoid external deps
    t = token.lower()
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"     # stories -> story
    if len(t) > 3 and t.endswith("ing"):
        return t[:-3]           # playing -> play
    if len(t) > 2 and t.endswith("ed"):
        return t[:-2]           # worked -> work
    if len(t) > 3 and t.endswith("es"):
        return t[:-2]           # boxes -> box
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]           # books -> book
    return t

def split_sentences(text: str) -> List[str]:
    # Split on . ! ? while keeping order; naive but serviceable
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def tokenize(text: str, cfg: PreprocessConfig) -> List[str]:
    if cfg.lowercase:
        text = text.lower()
    toks = [m.group(0) for m in _WORD_RE.finditer(text)]
    if cfg.remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if cfg.stemming:
        toks = [_simple_stem(t) for t in toks]
    return toks

def preprocess_text(text: str, title: Optional[str] = None, cfg: Optional[PreprocessConfig] = None) -> Document:
    cfg = cfg or PreprocessConfig()
    sents_raw = split_sentences(text)
    sentences = []
    for i, s in enumerate(sents_raw):
        tokens = tokenize(s, cfg)
        sentences.append(Sentence(idx=i, text=s, tokens=tokens))
    
    # Create document first
    doc = Document(title=title, raw_text=text, sentences=sentences)
    
    # Compute TF-IDF vectors for all sentences
    from .features import _compute_idf, _compute_tfidf_vector
    idf_scores = _compute_idf(doc, smooth_idf=True)
    
    # Update each sentence with its TF-IDF vector
    for sentence in doc.sentences:
        sentence.tf_idf_vector = _compute_tfidf_vector(sentence.tokens, idf_scores, tf_mode="sublinear")
    
    return doc
