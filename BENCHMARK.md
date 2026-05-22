# Data Ingestion & Vector Retrieval

A. Strategi Pemotongan Teks (Chunking)
Ini akan sangat memengaruhi apa yang direpresentasikan oleh vektor.

Ukuran (Size): 128, 256, 512, 1024 token.

Tumpang Tindih (Overlap): 0%, 10%, 25%.

Metode: Pemotongan statis (jumlah karakter fix) vs Pemotongan sintaksis (berhenti di titik/akhir paragraf) vs Semantic Chunking (berhenti saat topik berubah).

B. Representasi Vektor (Embedding Model)
Model embedding menentukan kualitas pemahaman semantik dari database.

Dimensi Vektor: Membandingkan model dimensi kecil (misal: 384 dimensi pada MiniLM) versus dimensi besar (misal: 1536 dimensi pada OpenAI atau 1024 pada BGE-M3).

Tipe Model: Model dense standar vs model multi-bahasa (jika dokumen Anda berbahasa Indonesia).

## Pandung Pengisian Bencmarking

## Hasil Benchmarking

| Ukuran Chunk (Chars) | Overlap (Chars) | Metode Chunking | Model Embedding | Top-K (Limit) | Filter Metadata | Hit Rate (Recall@K) | Rata-rata Latensi (ms) | Ukuran Index DB (MB) | Catatan / Insight RAG |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1000 | 200 | Hybrid (Header + Recursive) | BAAI/bge-m3 (Dense 1024-dim) | 5 | Ya (DOI) | - | - | - | *Baseline saat ini* |
| 500 | 100 | Hybrid (Header + Recursive) | BAAI/bge-m3 (Dense 1024-dim) | 10 | Ya (DOI) | - | - | - | *Eksperimen: Chunk kecil, Top-K besar* |
| 1000 | 200 | Recursive Statis | sentence-transformers/all-MiniLM-L6-v2 | 5 | Tidak | - | - | - | *Eksperimen: Model ringan tanpa filter* |

## Panduan Pengisian Benchmarking

Evaluasi Sistem *Retrieval* (Pengambilan Data) adalah nyawa dari arsitektur RAG. Berikut adalah penjelasan ringkas mengapa kolom-kolom metrik di atas sangat penting untuk dipantau dalam fase eksperimen Anda:

### 1. `Top-K (Limit)`
- **Definisi:** Jumlah *chunk* maksimal yang dikembalikan oleh Qdrant ke Gemini.
- **Insight:** Terkadang, menyetel *Top-K* ke angka 10 dengan ukuran *chunk* yang lebih kecil (misal 500 karakter) justru memberikan konteks yang jauh lebih beragam (berasal dari berbagai halaman) dibandingkan mengambil *Top-K* 3 dengan *chunk* raksasa.

### 2. `Hit Rate (Recall@K)`
- **Definisi:** Persentase keberhasilan sistem menemukan "*Chunk* yang mengandung jawaban yang benar" pada pencarian Top-K.
- **Insight:** Ini adalah metrik paling krusial. Anda bisa membuat 10 pasang pertanyaan-jawaban tes. Jika dari 10 pertanyaan tersebut Qdrant berhasil menemukan paragraf yang tepat sebanyak 8 kali di peringkat atas, maka *Hit Rate (Recall)* Anda adalah 80%. Semakin tinggi nilainya, semakin kecil risiko LLM berhalusinasi.

### 3. `Filter Metadata`
- **Definisi:** Apakah pencarian menggunakan *pre-filtering* (seperti membatasi pencarian hanya pada DOI atau *Section Header* tertentu)?
- **Insight:** Filter DOI bisa meningkatkan *Hit Rate* menjadi nyaris 100% secara instan karena sistem secara eksplisit "membuang" gangguan teks (*noise*) dari jurnal lain yang tidak relevan.

### 4. `Rata-rata Latensi (ms)`
- **Definisi:** Waktu komputasi komprehensif dari saat kueri dikirim hingga Qdrant mengembalikan hasilnya.
- **Insight:** Sangat krusial untuk lingkungan *Production*. Jika Anda beralih menggunakan model *embedding* raksasa, latensi pencarian bisa melambung tinggi. Harus diperhitungkan jika aplikasi akan diakses ratusan pengguna serentak.

### 5. `Ukuran Index DB (MB)`
- **Definisi:** Ukuran total folder *database* lokal (contoh: `qdrant_db/`) untuk jumlah kumpulan jurnal uji tertentu.
- **Insight:** Berhubungan langsung dengan konsumsi *Storage* dan *RAM/VRAM* di *Cloud*. Apakah mengorbankan akurasi sebesar 2% sepadan dengan penghematan penyimpanan sebesar 50%? Jawabannya dapat dievaluasi di sini.