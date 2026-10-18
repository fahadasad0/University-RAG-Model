# SIBAU Authorized RAG — class project

This Streamlit application answers questions from the uploaded **SIBAU Employee Handbook (2018)** using role-based retrieval and Groq's `openai/gpt-oss-20b` model.

## Included artifacts
- `data/chunks.jsonl` — 261 page-aware text chunks with source/page metadata and access tags.
- `artifacts/vectors_tfidf.npz` — sparse TF-IDF vectors for the chunks.
- `artifacts/tfidf_vectorizer.joblib` — fitted vectorizer used at query time.
- `artifacts/index_manifest.json` — index description, matrix dimensions and access-tag policy.
- `app.py` — Streamlit app with role-based access control and Groq API calls.

**Important embedding note:** the included vectors are TF-IDF lexical vectors, not neural dense embeddings. They are self-contained and can be used without a separate embedding API. This is a practical class-project baseline; if your rubric explicitly requires semantic dense embeddings, replace this index with a dense embedding model and rebuild both document/query embeddings with the same model.

## Demo accounts for a class presentation

These are **throwaway demonstration credentials**. Do not reuse them for real accounts. The app stores SHA-256 password hashes in Streamlit Secrets; usernames and roles are shown below for the presentation.

| Username | Demo password | Role |
|---|---|---|
| `student_demo` | `SIBAUstudent2026!` | `student` |
| `faculty_demo` | `SIBAUfaculty2026!` | `faculty` |
| `staff_demo` | `SIBAUstaff2026!` | `staff` |
| `hr_demo` | `SIBAUhr2026!` | `hr` |
| `admin_demo` | `SIBAUadmin2026!` | `admin` |

### Add the accounts to Streamlit Cloud

1. Push this project to GitHub.
2. Open your app in [Streamlit Community Cloud](https://share.streamlit.io/).
3. Open **App settings → Secrets**.
4. Copy the contents of `secrets.toml.example` into the Secrets editor.
5. Replace `gsk_REPLACE_WITH_YOUR_GROQ_KEY` with your own Groq API key. Never commit that key to GitHub.
6. Save the Secrets and reboot/redeploy the app.
7. Sign in with one of the demo username/password pairs above.

`secrets.toml.example` contains password hashes, not plaintext passwords. It is a template for the Streamlit Secrets editor; do **not** rename it to a committed `.streamlit/secrets.toml` or upload actual API keys to GitHub.

## Role model (editable)
- `student`: general handbook chunks.
- `faculty`, `staff`: general and employee-policy chunks.
- `hr`, `admin`: general, employee-policy, and restricted-HR-topic chunks.

The tags are a **demonstration policy**, not official SIBAU authorization rules. Review them with your instructor and data owner. Since this source is an employee handbook, the restricted tag currently limits selected compensation, benefits, loans, and personnel-file-related passages to `hr` and `admin`. This does not mean the public PDF itself is confidential; it demonstrates how access can be scoped.

## Security boundaries
- Retrieval filters chunks by role **before** similarity ranking and before sending excerpts to Groq.
- The UI is not the security boundary; the server-side role check is.
- Streamlit session-state login is suitable for a class prototype, not a production identity system. For real deployment use university SSO/OIDC, rate limiting, audit logging, secret rotation, and HTTPS.
- Do not upload private employee records unless you have authorization and an approved data-handling process.
- The app only supplies retrieved, authorized excerpts to the model; prompt instructions are not a substitute for authorization checks.
- The app does not fine-tune or train a model on the handbook. It retrieves chunks and uses them as context.

## Rebuild the index from the PDF
The prebuilt index is included, so deployment does not need the original PDF. To rebuild after changing the document, run `build_index.py` (requires PyMuPDF, scikit-learn, scipy, joblib, numpy). Review the generated access tags before deployment.
