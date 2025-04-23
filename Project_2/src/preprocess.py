import re

COMPANY_LIST = [
    "google", "apple", "microsoft", "amazon", "facebook", "tesla",
    "oracle", "ibm", "intel", "nvidia", "qualcomm", "sap",
    "salesforce", "uber", "airbnb", "twitter", "meta", "snap",
    "zoom", "palantir"
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def mask_text(text: str) -> str:
    t = text.replace("\n", " ").strip().lower()
    t = ''.join('[NUM]' if ch.isdigit() else ch for ch in t)
    for comp in COMPANY_LIST:
        t = re.sub(rf"\b{comp}\b", '[COMPANY]', t)
    return t
