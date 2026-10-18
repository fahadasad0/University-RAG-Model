"""Rebuild chunks and TF-IDF vector artifacts from Employee_Handbook_2018.pdf."""
import json, re
from pathlib import Path
import fitz, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

ROOT = Path(__file__).parent
PDF = ROOT / "Employee_Handbook_2018.pdf"
DATA = ROOT / "data"; ART = ROOT / "artifacts"
DATA.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)

def access_for(text, page_num):
    low = text.lower()
    restricted_terms = ["salary and benefits", "pay protection", "employee loan",
        "vehicle loan", "laptop loan", "provident fund", "gratuity", "health insurance",
        "earned leave encashment", "tuition fees", "house allotment", "hostels allotment",
        "personnel file", "pay fixation", "payroll", "salary grade"]
    if any(term in low for term in restricted_terms):
        return ["hr", "admin"], "restricted_hr"
    if 17 <= page_num <= 96:
        return ["faculty", "staff", "hr", "admin"], "employee_policy"
    return ["student", "faculty", "staff", "hr", "admin"], "general"

doc = fitz.open(PDF); chunks = []
for page_num, page in enumerate(doc, start=1):
    txt = re.sub(r"[ \t]+", " ", page.get_text("text"))
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    if not txt: continue
    start = 0
    while start < len(txt):
        end = min(len(txt), start + 1150)
        if end < len(txt):
            boundary = max(txt.rfind("\n", start + 575, end), txt.rfind(". ", start + 575, end))
            if boundary > start: end = boundary + 1
        body = txt[start:end].strip()
        if len(body) >= 80:
            roles, tag = access_for(body, page_num)
            chunks.append({"chunk_id": f"sibau-p{page_num:03d}-c{len(chunks)+1:05d}",
                "source": "SIBAU Employee Handbook", "source_file": PDF.name,
                "page_start": page_num, "page_end": page_num, "text": body,
                "access": {"allowed_roles": roles, "tag": tag}})
        if end >= len(txt): break
        start = max(start + 1, end - 180)
with open(DATA/"chunks.jsonl", "w", encoding="utf-8") as f:
    for c in chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")
vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1,2),
    max_features=50000, sublinear_tf=True, norm="l2",
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_'-]*\b")
matrix = vec.fit_transform([c["text"] for c in chunks])
joblib.dump(vec, ART/"tfidf_vectorizer.joblib", compress=3)
save_npz(ART/"vectors_tfidf.npz", matrix)
with open(ART/"index_manifest.json", "w", encoding="utf-8") as f:
    json.dump({"pages_processed": len(doc), "chunks": len(chunks),
      "vector_type": "TF-IDF sparse vectors; not dense neural embeddings",
      "matrix_shape": list(matrix.shape)}, f, indent=2)
print(f"Built {len(chunks)} chunks; matrix shape {matrix.shape}")
