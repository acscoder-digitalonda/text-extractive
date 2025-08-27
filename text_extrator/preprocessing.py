from __future__ import annotations
import re, base64, binascii, math
from dataclasses import dataclass
from typing import List, Optional
from .datatypes import Document, Sentence
from collections import Counter


RE_BASE64  = re.compile(r'^[A-Za-z0-9+/]{20,}={0,2}$')   # chuỗi dài kiểu base64
RE_HEX     = re.compile(r'^[0-9a-fA-F]{16,}$')           # hex dài (hash)
RE_UUID    = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$')
RE_URL     = re.compile(r'^(?:https?|ftp)://', re.I)

# 2) Entropy (bit/ký tự); chuỗi quá “ngẫu nhiên” thường là rác
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

# 3) Thử decode base64 và kiểm tra có phải dữ liệu nhị phân (nhiều byte không in được)
def looks_like_binary_after_b64(s: str) -> bool:
    # base64 phải có độ dài bội số 4 mới "sạch"; nhưng nhiều chuỗi thiếu padding
    if len(s) < 20 or len(s) % 4 != 0:
        return False
    try:
        raw = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return False
    if not raw:
        return False
    # Nếu >30% byte là không in được (ngoài \t\n\r) → khả năng cao là nhị phân
    nonprint = sum(1 for b in raw if b < 32 and b not in (9,10,13)) + sum(1 for b in raw if b == 127)
    return (nonprint / len(raw)) > 0.30

# 4) Quy tắc quyết định "noise"
def is_noise_token(tok: str) -> bool:
    if not tok:
        return True

    # Bỏ qua URL, email, @mention… tuỳ nhu cầu
    if RE_URL.search(tok):
        return False  # có thể giữ lại URL; đổi thành True nếu bạn muốn lọc

    # Loại base64/hex/uuid/hash dài
    if RE_UUID.match(tok):
        return True
    if RE_HEX.match(tok) and len(tok) >= 24:  # hex dài như SHA1/256…
        return True
    if RE_BASE64.match(tok) or looks_like_binary_after_b64(tok):
        return True

    # Loại chuỗi quá dài mà không có nguyên âm (tiếng Anh/VN) → có thể là rác
    vowels = set("aeiouyAEIOUYàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ")
    if len(tok) >= 20 and not any(ch in vowels for ch in tok):
        return True

    # Loại chuỗi có tỉ lệ chữ+số quá cao và entropy cao → giống random
    letters_digits = sum(ch.isalnum() for ch in tok)
    if len(tok) >= 16 and letters_digits / len(tok) > 0.95 and shannon_entropy(tok) > 4.0:
        return True

    return False

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
    toks = [t for t in toks if not is_noise_token(t)]
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
