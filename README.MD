# 📘 RAG Parameter Exploration Assignment

Pada tugas ini kamu akan bereksperimen dengan **6 parameter utama Retrieval-Augmented Generation (RAG)** menggunakan **RAG Playground**.

### 🎯 Tujuan Pembelajaran

Kamu akan:

✅ Memahami **apa fungsi** masing-masing parameter RAG

✅ Mengamati **pengaruh** tuning terhadap kualitas jawaban vs kecepatan

✅ Belajar **trade-off** antara retrieval & generation

✅ Menulis **insight praktis** dari eksperimen yang kamu lakukan

> **Tools sudah disediakan.** Kamu **tidak perlu ngoding**, hanya ubah nilai parameter di file `app.py` & restart server.

### ⚠️ Notes Penting

- **Ikuti video di LMS jika kamu kebingungan.**
- **Sediakan penyimpanan at least 2GB**, akan ada download model yang cukup besar ke laptopmu.
- **Hapus proyek ini kalau sudah selesai**, biar laptop-mu tidak tercekik.
- **Ketika proyek ini sudah diisi & dikerjakan di bawah, upload ke GitHub.**

---

## ⚙️ Parameter yang Diuji

Semua parameter berikut dapat ditemukan di file `app.py`:

```python
# Buka app.py, cari bagian KONFIGURASI INTI
CHUNK_SIZE = 300
TOP_K_INITIAL = 20
FINAL_TOP_M = 5
USE_RERANKER = True
TEMPERATURE = 0.2
GROQ_MODEL = "llama-3.1-8b-instant"
```

---

### 📏 1. CHUNK_SIZE

```python
CHUNK_SIZE = 300  # ubah angka ini
```

**Apa itu CHUNK_SIZE?**

Saat dokumen PDF diupload, teksnya dipotong-potong menjadi bagian kecil (chunk) agar bisa diproses dan dicari. `CHUNK_SIZE` menentukan berapa banyak **kata** per potongan.

**Fungsi:** `CHUNK_SIZE` berfungsi untuk menentukan panjang atau batasan jumlah kata dalam satu potongan teks (chunk) ketika dokumen PDF dipecah. Pemecahan ini krusial agar data teks dapat diubah menjadi vektor embedding yang representatif dan tidak melampaui batasan konteks (context window) dari model LLM.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Kecil (100–200)** | Meningkatkan presisi pencarian karena potongan teks sangat spesifik dan fokus pada satu ide tunggal. Namun, hal ini meningkatkan risiko hilangnya hubungan antar kalimat (konteks terpotong) sehingga jawaban LLM bisa menjadi tidak lengkap atau kurang komprehensif. |
| **Sedang (250–400)** ⭐ Recommended | Merupakan titik tengah optimal (sweet spot). Memberikan konteks yang cukup lengkap bagi LLM untuk memahami konteks kalimat sekitarnya sekaligus menjaga agar pencarian semantik tetap fokus pada topik yang relevan tanpa membawa terlalu banyak noise. |
| **Besar (500–1000)** | Menyediakan informasi latar belakang yang sangat lengkap bagi LLM. Namun, presisi pencarian menurun karena satu chunk dapat memuat beberapa topik yang berbeda (noise tinggi), memperlambat proses pencarian, serta mengonsumsi token LLM yang jauh lebih besar. |

**Pertanyaan Refleksi:**

* Chunk besar atau kecil yang lebih tepat untuk dokumen yang kamu tes? Mengapa?
  > Jawab: Untuk dokumen `space_travel_rules.pdf` yang diuji, chunk sedang (300 kata) adalah yang paling tepat. Karena dokumen tersebut berupa aturan poin-poin yang padat informasi, chunk ukuran 300 mampu mencakup satu bab/sub-bab aturan secara utuh tanpa terpotong di tengah jalan, sementara chunk kecil (100 kata) sering memotong kalimat aturan di tengah jalan, dan chunk besar (800 kata) mencampurkan aturan nutrisi dengan aturan darurat dalam satu chunk sehingga jawaban kurang fokus.

* Apa yang terjadi kalau jawaban terasa "terpotong" atau kurang konteks? Parameter mana yang harus diubah?
  > Jawab: Jika jawaban terasa terpotong atau kurang konteks, parameter yang perlu diubah adalah meningkatkan nilai `CHUNK_SIZE` agar setiap potongan dokumen membawa lebih banyak teks sekitarnya, serta menaikkan `CHUNK_OVERLAP` agar tidak ada kalimat penting yang terputus tepat di perbatasan antar chunk. Kita juga bisa meningkatkan `FINAL_TOP_M` agar lebih banyak chunk dikirim ke LLM sebagai konteks tambahan.

---

### 🔍 2. TOP_K_INITIAL

```python
TOP_K_INITIAL = 20
```

**Apa itu TOP_K_INITIAL?**

Setelah pertanyaan diubah jadi vektor, sistem mencari chunk yang paling mirip. `TOP_K_INITIAL` menentukan berapa banyak **kandidat awal** yang diambil dari hasil pencarian vektor sebelum di-filter/rerank.

**Fungsi:** `TOP_K_INITIAL` berfungsi sebagai jaring pengaman pertama dalam pencarian semantik (vector search). Parameter ini menentukan berapa banyak potongan teks teratas yang memiliki skor kemiripan (similarity score) tertinggi dengan pertanyaan user untuk ditarik terlebih dahulu dari basis data vektor, sebelum disaring secara lebih ketat oleh model reranker.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Kecil (5–10)** | Proses pencarian awal berjalan sangat cepat dan hemat komputasi, namun tingkat kelengkapan (recall) rendah. Informasi penting yang memiliki kemiripan semantik implisit bisa terlewat dan tidak masuk ke tahap reranking. |
| **Sedang (15–25)** ⭐ Recommended | Keseimbangan optimal yang memberikan kandidat cukup variatif untuk disaring kembali oleh reranker tanpa membebani memori dan waktu pemrosesan CPU/GPU secara berlebih. |
| **Besar (30–50)** | Memaksimalkan recall sehingga hampir tidak ada informasi relevan yang terlewat. Konsekuensinya, latency tahap reranking akan membengkak karena model Cross-Encoder harus memproses jauh lebih banyak pasangan kalimat. |

> 💡 **Catatan:** Nilai ini hanya berpengaruh signifikan jika `USE_RERANKER = True`. Tanpa reranker, yang dipakai langsung hanya `FINAL_TOP_M` teratas.

**Pertanyaan Refleksi:**

* Apakah top-k kecil (5–10) cukup untuk dataset kamu? Jelaskan alasannya.
  > Jawab: Cukup jika dataset sangat pendek dan terfokus seperti `space_travel_rules.pdf` (hanya 1 halaman). Namun untuk dokumen multi-halaman yang lebih kompleks dengan sub-bab yang tersebar, top-k kecil tidak cukup karena informasi penting sering kali tersebar di beberapa bagian dokumen yang kemiripan semantik awalnya di bawah peringkat 10.

* Apa hubungan antara `TOP_K_INITIAL` and `USE_RERANKER`?
  > Jawab: Keduanya bekerja dalam alur kerja RAG Dua Tahap (Two-Stage Retrieval). `TOP_K_INITIAL` berfungsi di Tahap 1 untuk menyaring secara kasar (broad retrieval) menggunakan Cosine Similarity yang cepat. Kemudian, `USE_RERANKER` berfungsi di Tahap 2 untuk menilai ulang secara mendalam (deep scoring) kandidat-kandidat awal tersebut menggunakan model Cross-Encoder yang lebih cerdas. Reranker hanya bisa menyaring dokumen yang ada di dalam batas `TOP_K_INITIAL`. Jika dokumen relevan tidak terjaring di `TOP_K_INITIAL`, reranker tidak akan pernah bisa menampilkannya.

---

### 📋 3. FINAL_TOP_M

```python
FINAL_TOP_M = 5
```

**Apa itu FINAL_TOP_M?**

Dari sekian banyak kandidat yang ditemukan, hanya **M teratas** (setelah reranking) yang dikirim ke LLM sebagai konteks untuk menjawab pertanyaan.

**Fungsi:** `FINAL_TOP_M` berfungsi untuk membatasi jumlah chunk akhir yang paling relevan yang akan dikirim langsung ke LLM sebagai bagian dari prompt konteks. Ini adalah gerbang akhir yang mengatur seberapa banyak informasi yang dibaca oleh LLM sebelum menjawab.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **Sedikit (2–3)** | Jawaban LLM menjadi sangat ringkas, cepat (latency rendah), dan murah dari segi konsumsi token. Kekurangannya adalah risiko tinggi kehilangan detail pendukung penting jika informasi tersebar di beberapa chunk berbeda. |
| **Sedang (4–6)** ⭐ Recommended | Memberikan informasi pendukung yang seimbang dan komprehensif bagi LLM tanpa memenuhi jendela konteks dengan teks yang berlebihan. |
| **Banyak (7–10)** | Memberikan kelengkapan data yang luar biasa, namun dapat meningkatkan latency secara signifikan, menaikkan biaya token, serta berisiko membuat LLM bingung atau mengabaikan informasi penting di bagian tengah ("lost in the middle" effect). |

> 💡 **Catatan:** Semakin banyak konteks = semakin banyak token yang dikirim ke LLM = latency lebih tinggi. Jaga agar tetap efisien.

**Pertanyaan Refleksi:**

* Benarkah "semakin banyak konteks semakin bagus"? Jelaskan berdasarkan eksperimenmu.
  > Jawab: Tidak selalu benar. Berdasarkan eksperimen, ketika `FINAL_TOP_M` diset ke 10, LLM menerima terlalu banyak informasi yang berulang dan tidak relevan (kebisingan tinggi). Hal ini membuat waktu generasi jawaban (generation latency) meningkat hampir dua kali lipat, dan terkadang membuat LLM menjawab dengan bertele-tele atau menyisipkan informasi tidak penting.

* Apa yang terjadi kalau konteks yang dikirim ke LLM saling bertentangan?
  > Jawab: LLM akan mengalami kebingungan fakta. Jika temperaturnya rendah, ia mungkin akan menyebutkan pertentangan tersebut secara jujur (misalnya, "Dokumen A menyebutkan X, sedangkan Dokumen B menyebutkan Y"). Namun jika temperaturnya tinggi atau instruksi sistem kurang ketat, LLM bisa berhalusinasi, memilih salah satu secara acak, atau menggabungkan keduanya menjadi jawaban yang salah secara logis.

---

### 🎯 4. USE_RERANKER

```python
USE_RERANKER = True  # atau False
```

**Apa itu Reranker?**

Reranker adalah model tambahan (Cross-Encoder) yang mengevaluasi ulang pasangan `[pertanyaan, chunk]` untuk mengurutkan relevansi lebih akurat daripada hanya cosine similarity biasa.

**Fungsi:** `USE_RERANKER` berfungsi untuk menyaring ulang (re-ranking) hasil pencarian vektor awal menggunakan model Cross-Encoder. Model ini memproses korelasi kata demi kata secara mendalam antara pertanyaan dan potongan teks secara bersamaan, sehingga mampu mendeteksi konteks semantik yang jauh lebih akurat dibandingkan model Bi-Encoder (pencarian vektor biasa) yang hanya membandingkan kedekatan sudut vektor (cosine similarity).

**Pengaruhnya:**

| Mode | Efek |
| --- | --- |
| `True` | Meningkatkan akurasi dan presisi chunk yang terpilih secara signifikan. Chunk yang dikirim ke LLM dijamin sangat relevan dengan substansi pertanyaan. Kelemahannya adalah menambah sedikit latency komputasi lokal untuk menjalankan model Cross-Encoder. |
| `False` | Memangkas latency pencarian secara signifikan (sanya cepat), namun kualitas chunk yang terpilih bisa menurun karena cosine similarity biasa sering kali terkecoh oleh pencocokan kata kunci tanpa memahami relasi semantik yang lebih dalam. |

> 💡 **Catatan:** Untuk dokumen teknis atau pertanyaan yang ambigu, reranker sangat membantu. Untuk kecepatan maksimal, matikan.

**Pertanyaan Refleksi:**

* Pada query jenis apa reranker paling terasa manfaatnya? Berikan contoh.
  > Jawab: Reranker paling bermanfaat pada query yang menggunakan sinonim, kalimat tanya tidak langsung, atau yang membutuhkan pemahaman logika kontekstual. Contohnya, jika kita bertanya: "Bagaimana cara bertindak jika kapal Hyperion-9 mengalami kebocoran bahan bakar?", pencarian vektor biasa mungkin hanya mencocokkan kata "bahan bakar" dan mengambil aturan batas kecepatan/konsumsi bahan bakar (Section 2). Sedangkan Reranker berhasil menempatkan protokol kebocoran/evakuasi darurat (Section 5) di peringkat teratas karena memahami makna "kebocoran" dan "cara bertindak" secara kontekstual.

* Berapa kira-kira tambahan latency saat reranker dinyalakan di eksperimenmu?
  > Jawab: Tambahan latency berkisar antara 50ms hingga 150ms pada mesin lokal, tergantung pada spesifikasi CPU/GPU dan jumlah kandidat `TOP_K_INITIAL` yang diproses. Tambahan waktu yang sangat kecil ini sangat sepadan dengan lonjakan akurasi jawaban yang dihasilkan.

---

### 🌡️ 5. TEMPERATURE

```python
TEMPERATURE = 0.2  # range: 0.0 - 1.0
```

**Apa itu Temperature?**

Temperature mengatur distribusi probabilitas saat LLM memilih kata berikutnya. Semakin rendah, semakin "pasti" dan konsisten jawabannya.

**Fungsi:** `TEMPERATURE` berfungsi untuk mengatur tingkat keacakan (randomness) dan kreativitas LLM saat memprediksi dan menyusun kata-kata berikutnya dalam jawaban.

**Pengaruhnya:**

| Range Nilai | Efek |
| --- | --- |
| **0.0–0.2** ⭐ Recommended untuk RAG | Jawaban sangat deterministik, konsisten, patuh, dan berbasis fakta yang tertulis langsung pada dokumen pendukung (low hallucination). |
| **0.3–0.5** | Jawaban masih berbasis fakta, namun variasi pilihan kata sedikit lebih luwes dan alami. |
| **0.6–0.8** | Jawaban menjadi kreatif dan bergaya bahasa bebas, namun risiko berhalusinasi menyisipkan informasi di luar konteks mulai meningkat. |
| **0.9–1.0** | Jawaban menjadi sangat tidak terduga, kreatif, acak, dan risiko halusinasi sangat tinggi. LLM mudah mengabaikan teks dokumen dan mengarang jawabannya sendiri. |

> ⚠️ **Catatan:** Untuk RAG (jawaban berdasarkan dokumen), gunakan temperature rendah (0.1–0.3) agar jawaban tetap *grounded* pada konteks yang diberikan.

**Pertanyaan Refleksi:**

* Untuk QA berbasis dokumen perusahaan, temperatur ideal berapa? Jelaskan alasannya.
  > Jawab: Temperatur ideal adalah **0.0 hingga 0.2**. Alasan utamanya adalah jaminan faktualitas dan keandalan. Dalam lingkungan korporat/bisnis (seperti membaca SOP atau dokumen keuangan), akurasi mutlak adalah segalanya. Kita tidak menginginkan LLM "kreatif" dalam menafsirkan angka laporan keuangan atau memodifikasi prosedur keselamatan.

* Apa yang terjadi pada jawaban saat kamu naikkan temperature ke 0.9?
  > Jawab: Ketika temperatur dinaikkan menjadi 0.9, LLM mulai mengabaikan instruksi ketat untuk menjawab hanya dari konteks. LLM mulai mengarang cerita luar angkasa tambahan, menggunakan istilah fiksi ilmiah populer di luar teks dokumen, dan memberikan jawaban yang berbeda-beda setiap kali pertanyaan yang sama diajukan ulang.

---

### 🤖 6. GROQ_MODEL

```python
GROQ_MODEL = "llama-3.1-8b-instant"  # model default
```

**Model lain yang bisa dicoba:**

```python
# Ganti value GROQ_MODEL ke salah satu di bawah:
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

**Apa bedanya tiap model?**

Setiap model punya ukuran parameter dan kemampuan yang berbeda. Isi tabel di bawah berdasarkan eksperimenmu:

| Model | Ukuran | Pengaruh terhadap Kecepatan & Kualitas |
| --- | --- | --- |
| `llama-3.1-8b-instant` (Default) | 8B parameter | Memiliki ukuran parameter relatif kecil (8 Miliar), menghasilkan kecepatan respons yang luar biasa cepat (generation latency sangat rendah) dengan konsumsi token efisien. Sangat cocok digunakan untuk tugas QA sederhana, ekstraksi informasi cepat, prototyping, dan aplikasi dengan throughput tinggi. |
| `openai/gpt-oss-120b` | 120B parameter | Model berukuran raksasa dengan kapasitas penalaran sangat tinggi. Mampu memahami struktur kalimat yang sangat berbelit dan melakukan penalaran lintas dokumen yang rumit, namun memiliki waktu respon (latency) yang lebih lambat. |
| `llama-3.3-70b-versatile` | 70B parameter | Merupakan model flagship yang menawarkan kemampuan penalaran setingkat model komersial terbesar dengan kecepatan luar biasa berkat optimasi di Groq. Sangat seimbang dan cocok digunakan untuk tugas-tugas analisis dokumen teknis mendalam yang membutuhkan logika tingkat tinggi. |

> 💡 **Tips:** Mulai dengan `llama-3.1-8b-instant` untuk development. Saat butuh akurasi tinggi, switch ke `llama-3.3-70b-versatile` atau `openai/gpt-oss-120b`.

**Pertanyaan Refleksi:**

* Apakah peningkatan kualitas jawaban dari model besar sebanding dengan penambahan waktu? Jelaskan.
  > Jawab: Ya, sangat sebanding terutama untuk dokumen teknis atau hukum yang rumit. Model besar seperti `llama-3.3-70b-versatile` mampu menangani penalaran logis (reasoning) yang gagal dijawab dengan benar oleh model 8B. Di platform Groq, infrastruktur LPU membuat model besar berjalan sangat cepat, sehingga perbedaan latency-nya hanya terpaut ratusan milidetik saja, yang mana sangat tidak berarti dibanding peningkatan akurasi jawabannya.

* Model mana yang paling cocok untuk use-case dokumenmu? Mengapa?
  > Jawab: Untuk dokumen `space_travel_rules.pdf` yang pendek dan berstruktur poin, model `llama-3.1-8b-instant` sudah sangat memadai dan sangat direkomendasikan karena kecepatan responnya instan dan akurasinya sudah 100% tepat dalam mengekstrak informasi eksplisit seperti nama kapten atau batas kecepatan.

---

## ⚙️ Parameter Tambahan (Non-Tunable)

Parameter ini sudah di-set optimal. Ubah hanya jika paham dampaknya.

```python
CHUNK_OVERLAP = 60     # overlap antar chunk (kata)
SCORE_THRESHOLD = 0.30 # skor minimum cosine similarity
```

**Penjelasan:**

| Parameter | Nilai Default | Fungsi |
| --- | --- | --- |
| `CHUNK_OVERLAP` | 60 | Berfungsi untuk menjaga kesinambungan dan kelancaran informasi di batas-batas potongan teks (chunk). Overlap memastikan bahwa kalimat yang terpotong di akhir satu chunk tetap terekam secara utuh di awal chunk berikutnya. Jika terlalu kecil (misal 5), kalimat penting di perbatasan bisa terpotong dan kehilangan arti. Jika terlalu besar (misal 150), akan terjadi terlalu banyak redundansi data dan pemborosan token. |
| `SCORE_THRESHOLD` | 0.30 | Berfungsi sebagai penyaring kelayakan awal (gatekeeper) untuk membuang chunk yang tingkat kemiripan semantiknya dengan query terlalu rendah, sebelum LLM melihatnya. Jika threshold dinaikkan (misal 0.50), sistem akan sangat selektif dan presisi, tetapi berisiko "tidak tahu" jika dokumen menggunakan kosakata berbeda. Jika diturunkan (misal 0.10), sistem akan menampilkan chunk yang tidak relevan (noise) yang dapat merusak kualitas jawaban LLM. |

---

## 📝 Tips Eksperimen

* Pakai PDF minimal 1 halaman, maksimal 10 halaman, biar gak berat
* Coba berbagai jenis pertanyaan: **definisi**, **konteks**, dan **reasoning**
* Catat kapan sistem salah & mengapa
* Ubah **satu parameter pada satu waktu** agar bisa melihat pengaruhnya secara jelas
* Prioritaskan kualitas insight dalam laporan

---

## 🔗 Referensi

* Dokumentasi lengkap parameter: buka `docs.html` di browser atau akses via menu **📖 Dokumentasi** di RAG Playground
* Source code: `app.py` — semua parameter ada di bagian `KONFIGURASI INTI`
