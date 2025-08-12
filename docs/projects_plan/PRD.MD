# Software Requirement Specification (SRS)
# Project: CourtSight

| Versi Dokumen | Tanggal | Penulis | Status |
| :--- | :--- | :--- | :--- |
| 1.0 | 12 Agustus 2025 | AI Assistant (Gemini) | Final |

---

## Daftar Isi
1.  [**Pendahuluan**](#1-pendahuluan)
    1.1. [Tujuan Proyek](#11-tujuan-proyek)
    1.2. [Ruang Lingkup Proyek](#12-ruang-lingkup-proyek)
    1.3. [Target Audiens & Pemangku Kepentingan](#13-target-audiens--pemangku-kepentingan)
    1.4. [Definisi, Akronim, dan Singkatan](#14-definisi-akronim-dan-singkatan)
2.  [**Deskripsi Keseluruhan**](#2-deskripsi-keseluruhan)
    2.1. [Konteks & Perspektif Produk](#21-konteks--perspektif-produk)
    2.2. [Ringkasan Fitur Utama](#22-ringkasan-fitur-utama)
    2.3. [Karakteristik Pengguna](#23-karakteristik-pengguna)
    2.4. [Asumsi dan Ketergantungan](#24-asumsi-dan-ketergantungan)
3.  [**Arsitektur Sistem & Tumpukan Teknologi**](#3-arsitektur-sistem--tumpukan-teknologi)
    3.1. [Gambaran Arsitektur Tingkat Tinggi](#31-gambaran-arsitektur-tingkat-tinggi)
    3.2. [Tumpukan Teknologi](#32-tumpukan-teknologi)
4.  [**Kebutuhan Fungsional**](#4-kebutuhan-fungsional)
    4.1. [Fitur 1: Mesin Pencari Berbasis AI untuk Putusan Mahkamah Agung](#41-fitur-1-mesin-pencari-berbasis-ai-untuk-putusan-mahkamah-agung)
    4.2. [Fitur 2: Chatbot AI untuk Hukum Mahkamah Agung](#42-fitur-2-chatbot-ai-untuk-hukum-mahkamah-agung)
    4.3. [Fitur 3: Asisten Suara AI untuk Pencatatan Notulensi Pengacara](#43-fitur-3-asisten-suara-ai-untuk-pencatatan-notulensi-pengacara)
5.  [**Kebutuhan Non-Fungsional**](#5-kebutuhan-non-fungsional)
    5.1. [Kinerja](#51-kinerja)
    5.2. [Skalabilitas](#52-skalabilitas)
    5.3. [Keamanan](#53-keamanan)
    5.4. [Ketersediaan & Keandalan](#54-ketersediaan--keandalan)
    5.5. [Kegunaan & Pengalaman Pengguna (UX)](#55-kegunaan--pengalaman-pengguna-ux)
    5.6. [Kepatuhan & Privasi Data](#56-kepatuhan--privasi-data)
6.  [**Lampiran**](#6-lampiran)
    6.1. [Diagram Sekuens](#61-diagram-sekuens)

---

## 1. Pendahuluan

### 1.1. Tujuan Proyek
Dokumen ini mendefinisikan persyaratan untuk proyek **CourtSight**, sebuah platform kecerdasan hukum berbasis AI. Visi utama CourtSight adalah untuk mendemokratisasi akses terhadap informasi hukum di Asia Tenggara dengan menyediakan alat yang intuitif untuk mencari, memahami, dan berinteraksi dengan data putusan Mahkamah Agung yang bersifat publik. Proyek ini bertujuan untuk mengubah dokumen hukum yang padat dan tidak terstruktur menjadi wawasan yang dapat dicari dan ditindaklanjuti, sehingga menjadikan keadilan lebih transparan dan inklusif.

### 1.2. Ruang Lingkup Proyek
Ruang lingkup proyek ini mencakup pengembangan platform web dan seluler dengan tiga fitur inti:
1.  **Mesin Pencari Semantik:** Pengindeksan dan pencarian putusan publik Mahkamah Agung dari berbagai yurisdiksi di Asia Tenggara.
2.  **Antarmuka Chatbot:** Interaksi percakapan untuk Q&A, analisis, dan ringkasan berbasis data hukum yang terindeks.
3.  **Asisten Suara:** Transkripsi dan analisis percakapan secara *real-time* untuk para profesional hukum, dengan integrasi ke basis data hukum.

Proyek ini akan dikembangkan secara bertahap (per fase) seperti yang dijelaskan dalam Kebutuhan Fungsional.

### 1.3. Target Audiens & Pemangku Kepentingan
* **Pengguna Utama (B2B & B2C):**
    * Firma Hukum
    * Pengacara Independen
    * Mahasiswa Hukum
    * Jurnalis Investigasi
    * Peneliti & Akademisi
* **Pemangku Kepentingan Internal:**
    * Tim Pengembang
    * Manajer Produk
    * Tim Bisnis & Pemasaran
    * Investor

### 1.4. Definisi, Akronim, dan Singkatan
* **AI:** Artificial Intelligence (Kecerdasan Buatan)
* **LLM:** Large Language Model
* **RAG:** Retrieval-Augmented Generation
* **STT:** Speech-to-Text
* **API:** Application Programming Interface
* **UI/UX:** User Interface / User Experience
* **PDPA:** Personal Data Protection Act
* **SRS:** Software Requirement Specification

---

## 2. Deskripsi Keseluruhan

### 2.1. Konteks & Perspektif Produk
CourtSight adalah produk *Software as a Service* (SaaS) mandiri yang akan di-host di Google Cloud Platform (GCP). Produk ini akan berinteraksi dengan berbagai layanan LLM eksternal (SEA Lion, SEALLM) dan akan menyajikan datanya melalui aplikasi web (Next.js) dan aplikasi seluler (untuk fitur asisten suara). Arsitekturnya dirancang berbasis layanan (*service-oriented*) untuk memastikan skalabilitas dan kemudahan pemeliharaan.

### 2.2. Ringkasan Fitur Utama
* **Pencarian Cerdas:** Memungkinkan pengguna menemukan informasi relevan dari jutaan halaman dokumen hukum menggunakan bahasa alami, bukan hanya kata kunci.
* **Analisis Otomatis:** Menghasilkan ringkasan, poin-poin kunci, dan analisis tren dari hasil pencarian atau percakapan.
* **Produktivitas Pengacara:** Mengotomatiskan tugas pencatatan notulensi dan memberikan wawasan hukum secara *real-time* selama pertemuan.

### 2.3. Karakteristik Pengguna
Pengguna CourtSight memiliki tingkat keahlian teknis yang bervariasi. Pengacara dan jurnalis mungkin tidak memiliki latar belakang teknis yang mendalam, sehingga antarmuka harus sangat intuitif. Sebaliknya, peneliti atau mahasiswa mungkin lebih terbiasa dengan fitur pencarian lanjutan. Platform harus melayani kedua spektrum pengguna ini.

### 2.4. Asumsi dan Ketergantungan
* **Asumsi:**
    * Data putusan pengadilan yang akan diindeks tersedia secara publik dan dapat diakses secara legal.
    * Pengguna memiliki koneksi internet yang stabil untuk mengakses platform.
* **Ketergantungan:**
    * Ketersediaan dan performa API dari layanan LLM pihak ketiga (SEA Lion, dll.).
    * Ketersediaan layanan Google Cloud Platform.
    * Kualitas model LLM dalam memahami konteks hukum spesifik di Asia Tenggara.

---

## 3. Arsitektur Sistem & Tumpukan Teknologi

### 3.1. Gambaran Arsitektur Tingkat Tinggi
Sistem CourtSight akan mengadopsi arsitektur berbasis layanan mikro (*microservices*).
1.  **Frontend:** Aplikasi web `Next.js` berfungsi sebagai titik masuk utama bagi pengguna untuk fitur pencarian dan chatbot. Aplikasi seluler akan dikembangkan untuk fitur Asisten Suara.
2.  **Backend (API Gateway):** Sebuah `FastAPI` akan berfungsi sebagai *gateway* utama, menerima permintaan dari frontend dan mengorkestrasikannya ke layanan-layanan di bawahnya.
3.  **Orkestrator AI:** `LangGraph` akan digunakan untuk mengelola alur kerja yang kompleks, terutama untuk agen AI pada fitur chatbot dan asisten suara.
4.  **Layanan Khusus:**
    * **Embedding Service:** Layanan terpisah untuk mengubah teks menjadi vektor.
    * **LLM Service:** Wrapper untuk berinteraksi dengan berbagai model LLM.
    * **STT Engine:** Layanan untuk memproses transkripsi audio.
5.  **Lapisan Data:**
    * `PostgreSQL` akan menyimpan data terstruktur pengguna dan metadata dokumen.
    * `pgvector` (ekstensi PostgreSQL) akan digunakan sebagai *Vector Database* untuk pencarian kemiripan.

### 3.2. Tumpukan Teknologi

| Kategori | Teknologi | Tujuan |
| :--- | :--- | :--- |
| **Layanan LLM** | SEA Lion LLM, SEALLM, MERaLiON | Pemahaman bahasa, generasi teks, analisis audio |
| **Backend** | LangGraph, FastAPI | Orkestrasi agen AI, penyediaan API |
| **Database** | PostgreSQL, pgvector | Penyimpanan data relasional dan pencarian vektor |
| **Frontend** | Next.js, Tailwind CSS | Pembangunan antarmuka web yang modern dan responsif |
| **Hosting & Cloud**| Google Cloud Platform (GCP) | Infrastruktur, komputasi, dan penyimpanan |
| **Pengujian & Evaluasi** | RAGAS | Evaluasi kualitas dan akurasi sistem RAG |

---

## 4. Kebutuhan Fungsional

Berikut adalah rincian kebutuhan fungsional yang dipetakan langsung dari diagram sekuens yang disediakan.

### 4.1. Fitur 1: Mesin Pencari Berbasis AI untuk Putusan Mahkamah Agung
*Referensi Diagram: `AIBasedSearchEngineforSupremeCourt.jpg`*

| ID | Fase | Langkah dalam Diagram | Deskripsi Fungsional | Komponen Terlibat | Data / Objek |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1.1** | 1 | `User` -> `Web App (Frontend)` | Pengguna memasukkan kueri pencarian (dalam bahasa alami) pada kolom pencarian di antarmuka web. | Web App (Frontend) | `search_query` |
| **F1.2** | 1 | `Web App` -> `Backend API` | Frontend mengirimkan kueri pencarian dan parameter lainnya (opsi langganan, filter awal) ke backend melalui panggilan API. | Web App, Backend API | `search(query, options)` |
| **F1.3** | 1 | `Backend API` -> `Embedding Service` | Backend API meminta Layanan Embedding untuk mengubah kueri teks pengguna menjadi representasi numerik (vektor). | Backend API, Embedding Service | `query_text` |
| **F1.4** | 1 | `Embedding Service` -> `Backend API` | Layanan Embedding mengembalikan vektor dari kueri tersebut. | Embedding Service, Backend API | `query_vector` |
| **F1.5** | 1 | `Backend API` -> `Vector DB Store` | Backend melakukan pencarian kemiripan di database vektor menggunakan `query_vector` untuk menemukan dokumen yang relevan. | Backend API, Vector DB (pgvector) | `similarity_search(vector, top_k)` |
| **F1.6** | 1 | `Vector DB Store` -> `Backend API` | Database vektor mengembalikan daftar ID dokumen yang relevan beserta skor kemiripannya. | Vector DB, Backend API | `retrieved_docs`, `scores` |
| **F1.7** | 1 | `Backend API` -> `Doc Store` | Backend mengambil konten lengkap dari dokumen yang relevan dari penyimpanan dokumen utama (misalnya, PostgreSQL) berdasarkan ID. | Backend API, Doc Store (PostgreSQL) | `fetch_full_text(doc_ids)` |
| **F1.8** | 2 | `Backend API` -> `LLM Service` | Backend menyusun *prompt* yang berisi kueri pengguna dan konteks dari dokumen yang ditemukan, lalu mengirimkannya ke LLM untuk dibuatkan ringkasan dan analisis. | Backend API, LLM Service | `Prompt(query, context_docs)` |
| **F1.9** | 2 | `LLM Service` -> `Backend API` | LLM mengembalikan jawaban terstruktur yang berisi ringkasan, analisis, dan sitasi yang diekstrak. | LLM Service, Backend API | `structured_answer` |
| **F1.10**| 3 | `Backend API` -> `Validation/Citation Checker` | (Opsional) Sistem mengirimkan sitasi yang dihasilkan LLM ke modul validasi untuk diperiksa kebenarannya terhadap dokumen sumber. | Backend API, Validation Checker | `citations_to_validate` |
| **F1.11**| 3 | `Validation Checker` -> `Backend API` | Modul validasi mengembalikan hasil verifikasi, menandai sitasi yang valid atau tidak konsisten. | Validation Checker, Backend API | `verification_results` |
| **F1.12**| 1 | `Backend API` -> `Web App` | Backend mengirimkan respons akhir ke frontend. Fase 1: Daftar link. Fase 2: Tambah ringkasan & poin kunci. Fase 3: Tambah status validasi. | Backend API, Web App | `response(summary, points, links)` |
| **F1.13**| 1 | `Web App` -> `User` | Frontend menampilkan hasil pencarian dengan format yang mudah dibaca, termasuk ringkasan, poin kunci, dan tautan ke sumber asli. | Web App (Frontend) | Tampilan Antarmuka (UI) |

### 4.2. Fitur 2: Chatbot AI untuk Hukum Mahkamah Agung
*Referensi Diagram: `AIChatbotforSupremeCourtLaw.jpg`*

| ID | Fase | Langkah dalam Diagram | Deskripsi Fungsional | Komponen Terlibat | Data / Objek |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F2.1** | 1 | `User` -> `Chat UI` | Pengguna mengetik pesan atau pertanyaan di antarmuka chat. | Chat UI (Frontend) | `chat_message` |
| **F2.2** | 1 | `Chat UI` -> `Agent Executor` | Antarmuka chat mengirimkan pesan pengguna beserta riwayat percakapan (jika ada) ke *Agent Executor* di backend. | Chat UI, Agent Executor (LangGraph) | `POST(message, history)` |
| **F2.3** | 3 | `Agent Executor` -> `LLM Service` | Agent mengevaluasi maksud (intent) dari pesan pengguna untuk merencanakan tindakan selanjutnya (misalnya, "ini adalah pertanyaan yang butuh pencarian" atau "ini hanya sapaan"). | Agent Executor, LLM Service | `Evaluate_intent(history)` |
| **F2.4** | 3 | `LLM Service` -> `Agent Executor` | LLM mengembalikan rencana tindakan, misalnya, `Action: {"tool": "search", "query": "..."}`. | LLM Service, Agent Executor | `Action_plan` |
| **F2.5** | 1 | `Agent Executor` -> `Embedding Service` & `Vector DB` | (Sebagai bagian dari tindakan "search") Agent melakukan proses pencarian dokumen relevan menggunakan RAG (Retrieval-Augmented Generation) seperti pada Fitur 1. | Agent Executor, Embedding, Vector DB | `search_query` |
| **F2.6** | 2 | `Agent Executor` -> `LLM Service` (Loop) | Agent mampu melakukan pencarian lanjutan atau memperbaiki kueri berdasarkan konteks percakapan untuk menjawab pertanyaan susulan. | Agent Executor, LLM Service | `refine_query(history, context)` |
| **F2.7** | 1 | `Agent Executor` -> `LLM Service` | Setelah mendapatkan konteks dari dokumen, Agent meminta LLM untuk menyusun draf jawaban berdasarkan konteks tersebut. | Agent Executor, LLM Service | `Prompt(question, context)` |
| **F2.8** | 2 | `LLM Service` -> `Agent Executor` | LLM mengembalikan draf jawaban yang terstruktur, lengkap dengan sitasi dari dokumen konteks. | LLM Service, Agent Executor | `draft_answer` |
| **F2.9** | 3 | `Agent Executor` -> `Validation/Citation Guard` | (Opsional) Agent mengirimkan draf jawaban ke modul validasi untuk memeriksa konsistensi dan keakuratan sitasi. | Agent Executor, Validation Guard | `validate(draft_answer)` |
| **F2.10**| 3 | `Validation Guard` -> `Agent Executor` | Modul validasi mengembalikan hasil pengecekan dan potensi koreksi minor pada jawaban. | Validation Guard, Agent Executor | `checked_answer` |
| **F2.11**| 1 | `Agent Executor` -> `Chat UI` | Agent mengirimkan respons final dalam format JSON ke antarmuka chat. | Agent Executor, Chat UI | `JSON(answer, sources, links)` |
| **F2.12**| 1 | `Chat UI` -> `User` | Antarmuka menampilkan jawaban dari chatbot kepada pengguna, beserta sumber yang bisa diklik. | Chat UI (Frontend) | Tampilan Antarmuka (UI) |

### 4.3. Fitur 3: Asisten Suara AI untuk Pencatatan Notulensi Pengacara
*Referensi Diagram: `AIVoiceAssistantforLawyersNote-Taking.jpg`*

| ID | Fase | Langkah dalam Diagram | Deskripsi Fungsional | Komponen Terlibat | Data / Objek |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F3.1** | 1 | `Lawyer` -> `Voice Assistant App` | Pengguna (pengacara) menekan tombol untuk memulai sesi rekaman suara di aplikasi. | Voice Assistant App (Frontend) | Aksi `Start Recording` |
| **F3.2** | 1 | `App` -> `Audio Stream (realtime)` | Aplikasi mulai mengirimkan aliran data audio secara *real-time* (atau dalam potongan/chunk) ke backend. | Voice Assistant App, Backend Service | `audio_chunk` |
| **F3.3** | 1 | `Audio Stream` -> `STT Engine (Whisper)` | Layanan backend meneruskan `audio_chunk` ke Mesin Speech-to-Text (STT) untuk ditranskripsi. | Backend Service, STT Engine | `audio_chunk` |
| **F3.4** | 1 | `STT Engine` -> `Agent Executor` | Mesin STT mengembalikan hasil transkripsi (bisa parsial atau final) ke Agent Executor. | STT Engine, Agent Executor | `transcript_text` |
| **F3.5** | 2 | `Agent Executor` -> `LLM Service` | Setelah sejumlah teks terkumpul (atau di akhir sesi), Agent meminta LLM untuk membuat ringkasan dari transkripsi. | Agent Executor, LLM Service | `Prompt("Summarize:", transcript)` |
| **F3.6** | 2 | `LLM Service` -> `Agent Executor` | LLM mengembalikan draf ringkasan dan poin-poin kunci dari percakapan. | LLM Service, Agent Executor | `summary`, `key_points` |
| **F3.7** | 3 | `Agent Executor` -> `LLM Service` | Agent secara proaktif menganalisis transkripsi untuk mengidentifikasi pertanyaan atau topik hukum yang muncul dalam percakapan. | Agent Executor, LLM Service | `Prompt("Find legal questions in:")` |
| **F3.8** | 3 | `LLM Service` -> `Agent Executor` | LLM mengidentifikasi dan mengembalikan kueri hukum yang terdeteksi dari percakapan. | LLM Service, Agent Executor | `identified_legal_query` |
| **F3.9** | 3 | `Agent Executor` -> `Vector DB` / `Doc Store`| Agent menggunakan kueri yang teridentifikasi untuk secara otomatis mencari putusan hukum yang relevan di database (alur sama seperti F1.5 - F1.7). | Agent Executor, Vector DB, Doc Store | `search_query` |
| **F3.10**| 3 | `Agent Executor` -> `LLM Service` | Agent meminta LLM untuk mensintesis wawasan dari putusan yang ditemukan terkait dengan konteks percakapan. | Agent Executor, LLM Service | `Prompt(context, related_cases)` |
| **F3.11**| 1 | `Agent Executor` -> `Voice Assistant App` | Backend mengirimkan hasil akhir ke aplikasi frontend. Fase 1: Hanya transkripsi. Fase 2: Transkripsi + ringkasan. Fase 3: Transkripsi + ringkasan + wawasan hukum proaktif. | Agent Executor, Voice Assistant App | `JSON(transcript, summary, insights)` |
| **F3.12**| 1 | `Voice Assistant App` -> `Lawyer`| Aplikasi menampilkan semua informasi yang relevan kepada pengacara di antarmuka yang terstruktur. | Voice Assistant App | Tampilan Antarmuka (UI) |

---

## 5. Kebutuhan Non-Fungsional

### 5.1. Kinerja
* **Waktu Respons Pencarian:** Hasil pencarian awal (daftar tautan) harus ditampilkan dalam < 3 detik untuk 95% kueri.
* **Waktu Respons Chatbot:** Jawaban dari chatbot harus muncul dalam < 5 detik. Jawaban yang memerlukan rencana kompleks dapat memakan waktu hingga 10 detik.
* **Latensi Transkripsi:** Latensi antara ucapan dan penampilan teks di layar untuk fitur asisten suara harus < 1 detik untuk pengalaman *real-time*.

### 5.2. Skalabilitas
* Sistem harus mampu menangani 1.000 pengguna konkuren pada peluncuran awal dan dapat diskalakan secara horizontal untuk menangani pertumbuhan pengguna.
* Basis data (relasional dan vektor) harus mampu menangani puluhan juta dokumen putusan tanpa degradasi kinerja yang signifikan.

### 5.3. Keamanan
* **Enkripsi Data:** Semua data, terutama transkripsi suara dan data pengguna, harus dienkripsi saat transit (menggunakan TLS 1.3) dan saat disimpan (menggunakan enkripsi AES-256).
* **Autentikasi & Otorisasi:** Sistem harus menerapkan autentikasi pengguna yang kuat (misalnya, OAuth 2.0) dan kontrol akses berbasis peran (RBAC) untuk data sensitif.
* **Perlindungan:** Aplikasi harus dilindungi dari 10 kerentanan teratas OWASP, termasuk XSS, CSRF, dan SQL Injection.

### 5.4. Ketersediaan & Keandalan
* Target *uptime* untuk semua layanan yang menghadap pengguna adalah 99.5%.
* Harus ada strategi pencadangan (backup) basis data secara berkala dan rencana pemulihan bencana (disaster recovery plan).

### 5.5. Kegunaan & Pengalaman Pengguna (UX)
* Antarmuka harus bersih, modern, dan intuitif, meminimalkan kurva belajar bagi pengguna non-teknis.
* Platform web harus sepenuhnya responsif dan dapat diakses dengan baik di berbagai ukuran layar (desktop, tablet, seluler).
* Sistem harus menyediakan umpan balik yang jelas kepada pengguna (misalnya, status loading, pesan error yang informatif).

### 5.6. Kepatuhan & Privasi Data
* Platform harus mematuhi peraturan privasi data yang relevan di yurisdiksi target utama (misalnya, PDPA Singapura, UU PDP Indonesia).
* Harus ada pemisahan yang jelas antara data publik (putusan pengadilan) dan data pribadi pengguna (transkripsi, riwayat pencarian).
* Kebijakan Privasi dan Syarat & Ketentuan harus jelas dan mudah diakses oleh pengguna.

---

## 6. Lampiran

### 6.1. Diagram Sekuens
Analisis dan pemetaan kebutuhan fungsional dalam dokumen ini didasarkan pada tiga diagram sekuens yang disediakan:
1.  `AIBasedSearchEngineforSupremeCourt.jpg`
2.  `AIChatbotforSupremeCourtLaw.jpg`
3.  `AIVoiceAssistantforLawyersNote-Taking.jpg`

Diagram-diagram ini berfungsi sebagai referensi visual utama untuk alur kerja sistem dan interaksi antar komponen.