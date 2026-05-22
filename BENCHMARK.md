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

C. Variasi Tipe Query (Query Diversity)
Untuk mengevaluasi ketangguhan sistem pencarian secara komprehensif, pengujian (metrik *Hit Rate*) harus menggunakan variasi kueri berikut:

1. **Factoid/Simple Query:** Pertanyaan langsung (Misal: "Berapa dimensi dari model embedding BGE-M3?").
2. **Reasoning/Complex Query:** Pertanyaan yang membutuhkan sintesis konsep (Misal: "Mengapa IVF-PQ lebih hemat memori dibandingkan HNSW?").
3. **Paraphrased/Semantic Query:** Pertanyaan yang sengaja TIDAK menggunakan istilah yang ada di dalam teks, tetapi maknanya sama. Ini adalah ujian sesungguhnya bagi sebuah *Vector Database*.
4. **Conversational/Noisy Query:** Pertanyaan dengan gaya bahasa kasual, tidak baku, atau mengandung sedikit *typo*, menyerupai ketikan pengguna asli di dunia nyata.

## Hasil Benchmarking

| Ukuran Chunk | Overlap | Metode Chunking | Model Embedding | Dukungan Bahasa | Tipe Query Uji | Top-K | Filter Metadata | Hit Rate | Latensi | Ukuran Index DB | Catatan |
|:---:|:---:|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1000 | 200 | Hybrid | BAAI/bge-m3 | Multi-bahasa (>100) | Semantic/Paraphrased | 5 | Ya (DOI) | - | - | - | *BGE-M3: Juara untuk kueri lintas bahasa* |
| 500 | 100 | Hybrid | BAAI/bge-m3 | Multi-bahasa (>100) | Reasoning/Complex | 10 | Ya (DOI) | - | - | - | *Eksperimen: Chunk kecil, Top-K besar* |
| 1000 | 200 | Statis | sentence-transformers/all-MiniLM-L6-v2 | Hanya Inggris | Factoid | 5 | Tidak | - | - | - | *MiniLM: Sangat cepat tapi buruk di bahasa Indonesia* |
| 1000 | 200 | Semantic | nomic-ai/nomic-embed-text-v1.5 | Mayoritas Inggris | Conversational | 5 | Ya (DOI) | - | - | - | *Nomic: Konteks sangat panjang* |

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

### 6. `Dukungan Bahasa (Language Support)`
- **Definisi:** Kemampuan model untuk memetakan bahasa yang berbeda ke dalam ruang semantik yang berdekatan (*Cross-Lingual Information Retrieval*).
- **Insight:** Sangat krusial jika *database* Anda berisi jurnal Bahasa Inggris, tetapi *user* memasukkan pencarian (*query*) dalam Bahasa Indonesia. Model *Multi-bahasa* seperti BGE-M3 dapat mempertemukan *query* Bahasa Indonesia dengan teks jurnal Bahasa Inggris secara cerdas, sedangkan model *Hanya Inggris* (seperti MiniLM) akan gagal total dalam skenario lintas-bahasa ini.
