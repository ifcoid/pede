import os
import sys
import httpx
from core.vector_store import VectorStore

# 1. Pastikan API Key tersedia di environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Fallback untuk Google Colab
if not GEMINI_API_KEY:
    try:
        from google.colab import userdata
        GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    except (ImportError, Exception):
        pass

if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY tidak ditemukan di environment variables ataupun Google Colab userdata.")
    print("Cara set di Windows (PowerShell): $env:GEMINI_API_KEY=\"API_KEY_ANDA\"")
    sys.exit(1)

# 2. Pengaturan Pencarian
QUERY = "Apa hasil eksperimen utamanya dan berapa persen akurasinya?"
DOI_TARGET = "10.1016/j.inpa.2026.02.006"
LIMIT_CHUNKS = 5

print(f"[SEARCH] Mencari jawaban di Qdrant untuk DOI: {DOI_TARGET}")
print(f"[QUERY]  Pertanyaan: {QUERY}\n")

# 3. Ambil teks dari Qdrant Vector Database
vector_store = VectorStore()
results = vector_store.search(query=QUERY, n_results=LIMIT_CHUNKS, doi_filter=DOI_TARGET)

if not results:
    print("[ERROR] Tidak ada teks yang ditemukan di Qdrant untuk DOI tersebut.")
    sys.exit(1)

# 4. Rangkai Prompt RAG
context_texts = []
for i, r in enumerate(results, 1):
    meta = r["metadata"]
    context_texts.append(f"--- Teks {i} (Bagian: {meta.get('section_hierarchy', 'Unknown')}) ---\n{r['content']}\n")

combined_context = "\n".join(context_texts)

system_prompt = f"""Anda adalah asisten AI akademik yang ahli.
Tugas Anda adalah menjawab pertanyaan user HANYA berdasarkan teks konteks jurnal yang diberikan di bawah ini.
Jika jawabannya tidak ada di dalam teks, katakan "Informasi tidak ditemukan di dalam dokumen".
Jangan pernah mengarang jawaban (halusinasi).

KONTEKS JURNAL:
{combined_context}
"""

print("[INFO] Konteks berhasil dikumpulkan. Mengirim ke Gemini LLM...")

# 5. Kirim ke Gemini API (menggunakan HTTPX)
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

payload = {
    "contents": [
        {
            "parts": [
                {"text": f"{system_prompt}\n\nPertanyaan User: {QUERY}"}
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.2 # Temperature rendah agar jawaban faktual
    }
}

try:
    response = httpx.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    
    # Ambil teks jawaban Gemini
    answer = data["candidates"][0]["content"]["parts"][0]["text"]
    
    print("\n" + "="*50)
    print("JAWABAN GEMINI (RAG):")
    print("="*50)
    print(answer)
    print("="*50)
    
except Exception as e:
    print(f"\n[ERROR] Gagal menghubungi Gemini API: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"Detail: {e.response.text}")
