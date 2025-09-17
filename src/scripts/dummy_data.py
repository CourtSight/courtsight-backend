#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator & Loader: Dummy Kasus Narkoba untuk ParentChildRetriever

- Membuat N dokumen  kasus narkotika dengan variasi:
  * Peran: pemakai / pemilik (possession) / kurir / penjual / pengendali
  * Zat: sabu, ganja, heroin, ekstasi
  * Barang bukti: gram / butir (ekstasi)
  * Pengadilan & tahun, nomor perkara 
  * Heuristik rekomendasi (rehabilitasi / rentang pidana) ***DUMMY***

- Memasukkan dokumen ke ParentChildRetriever (child/parent chunking)
- Menjalankan uji retrieval dengan 5 query (termasuk chat ala WhatsApp)

DISCLAIMER: DOKUMEN DUMMY untuk uji teknis RAG. BUKAN nasihat hukum.
"""

import asyncio
import sys
import argparse
import random
from datetime import datetime
from typing import List, Dict, Tuple

# Tambahkan path app
sys.path.append('/code')

from langchain_core.documents import Document
from app.services.retrieval.parent_child import ParentChildRetriever
from app.services.llm_service import get_llm_service
from app.services.rag.chains import create_rag_chains


# ============== Fixed docs (ringkasan/faq/pedoman) ==============
FIXED_DOCS = [
    {
        "title": " Ringkasan UU Narkotika – Struktur & Definisi",
        "content": """
Ringkasan UU Narkotika (UU 35/2009) – Struktur & Definisi (disederhanakan):
- Pemakai (penyalahguna) → Pasal 127 (rehabilitasi dapat dipertimbangkan via assessment).
- Memiliki/menyimpan/menguasai → Pasal 112 (ancaman : 4–12 tahun; bergantung BB).
- Menjual/menjadi perantara/mengedarkan → Pasal 114 (ancaman : 5–20 tahun; BB/peran/jaringan).
- Semua angka/parameter di sini sengaja disederhanakan untuk uji RAG teknis.
""",
        "metadata": {
            "source": "undang_undang_",
            "nomor": "35/2009",
            "kategori": "narkotika",
            "tags": ["UU", "ringkasan", "struktur", "narkotika"],
        },
    },
    {
        "title": " Pasal 112 – Memiliki/Menyimpan/Menguasai (Ringkas)",
        "content": """
Pasal 112 (disederhanakan):
- Perbuatan: memiliki/menyimpan/menguasai Narkotika Gol. I bukan tanaman.
- Ancaman tentang ancaman pidana narkotika (sangat disederhanakan): 4–12 tahun.
- Faktor: berat barang bukti, peran, keadaan meringankan/memberatkan.
""",
        "metadata": {
            "source": "undang_undang_",
            "pasal": "112",
            "kategori": "narkotika_possession",
            "tags": ["pasal_112", "possession"],
        },
    },
    {
        "title": " Pasal 114 – Menjual/Perantara/Mengedarkan (Ringkas)",
        "content": """
Pasal 114 (disederhanakan):
- Perbuatan: menawarkan untuk dijual, menjual, membeli, perantara, mengedarkan Narkotika Gol. I.
- Ancaman tentang ancaman pidana narkotika (sangat disederhanakan): 5–20 tahun.
- Variabel penting: berat barang bukti, peran pelaku (kurir/penjual/pengendali), jaringan.
""",
        "metadata": {
            "source": "undang_undang_",
            "pasal": "114",
            "kategori": "narkotika_peredaran",
            "tags": ["pasal_114", "peredaran", "perantara"],
        },
    },
    {
        "title": " Pasal 127 – Penyalahguna (Pemakai) & Rehabilitasi",
        "content": """
Pasal 127 (disederhanakan):
- Penyalahguna (Gol. I/II/III).
- Untuk Gol. I, ancaman maksimum tentang ancaman pidana narkotika (sangat disederhanakan) sampai 4 tahun.
- Rehabilitasi: berdasarkan assessment (Tim Terpadu) bila kriteria pengguna/pecandu terpenuhi dan tidak terbukti peredaran.
""",
        "metadata": {
            "source": "undang_undang_",
            "pasal": "127",
            "kategori": "penyalahguna_rehabilitasi",
            "tags": ["pasal_127", "rehabilitasi", "pemakai"],
        },
    },
    {
        "title": " Pedoman Assessment & Rehabilitasi (Ringkas)",
        "content": """
Pedoman Assessment & Rehabilitasi (ringkas):
- Tujuan: memilah pengguna/pecandu vs peredaran.
- Proses: permohonan asesmen → asesmen medis & sosial → rekomendasi → monitoring.
- Faktor: hasil tes, riwayat, kondisi psiko-sosial, berat BB, bukti peredaran.
""",
        "metadata": {
            "source": "pedoman_",
            "instansi": "BNN",
            "kategori": "rehabilitasi",
            "tags": ["assessment", "rehabilitasi"],
        },
    },
    {
        "title": " FAQ ‘Berapa tahun hukuman narkoba?’",
        "content": """
FAQ singkat tentang ancaman pidana narkotika (sangat disederhanakan):
- Pemakai (Pasal 127): dapat dipertimbangkan rehabilitasi; pidana maksimum tentang ancaman pidana narkotika (sangat disederhanakan) s.d. 4 tahun.
- Memiliki (Pasal 112): 4–12 tahun tentang ancaman pidana narkotika (sangat disederhanakan).
- Peredaran/perantara (Pasal 114): 5–20 tahun tentang ancaman pidana narkotika (sangat disederhanakan).
""",
        "metadata": {
            "source": "faq_",
            "kategori": "faq_hukuman",
            "tags": ["berapa_tahun", "ancaman_pidana", "rentang_hukuman"],
        },
    },
    {
        "title": " Template Analisis Cepat Kasus Narkoba (Router Intent)",
        "content": """
Template Analisis Cepat:
- Input: jenis perbuatan (pemakai/possession/peredaran), berat BB, peran.
- Rujukan: 127 (pemakai), 112 (possession), 114 (peredaran).
- Output: daftar pasal relevan + rentang  + opsi rehabilitasi bila memenuhi.
""",
        "metadata": {
            "source": "template_",
            "kategori": "router_intent",
            "tags": ["router", "analisis_cepat"],
        },
    },
    {
        "title": " Saran & Strategi (Non-Nasihat Hukum) – Komunikasi Klien",
        "content": """
Saran umum tentang ancaman pidana narkotika (sangat disederhanakan):
1) Kumpulkan dokumen asesmen, tes lab, riwayat, dukungan keluarga.
2) Kooperatif; ikut program konseling/rehabilitasi bila tepat.
3) Tegaskan peran: pemakai vs peredaran.
4) Dokumentasikan BB dan asal-usul (sesuai hukum acara).
5) Mohon asesmen Tim Terpadu bila indikasi penyalahguna/pecandu.
""",
        "metadata": {
            "source": "saran_",
            "kategori": "saran_umum",
            "tags": ["saran", "strategi"],
        },
    },
]


# ============== Generator kasus sintetis tentang ancaman pidana narkotika (sangat disederhanakan) ==============
COURTS = ["PN.JKT", "PN.SBY", "PN.BDG", "PN.DPS", "PN.MDN", "PN.SMG", "PN.MKS", "PN.BJM"]
ROLES = ["pemakai", "pemilik", "kurir", "penjual", "pengendali"]
SUBSTANCES = ["sabu", "ganja", "heroin", "ekstasi"]

def _bb_unit(substance: str) -> str:
    return "butir" if substance == "ekstasi" else "gram"

def _rand_bb(substance: str, role: str) -> float:
    """Berat/barang bukti  (gram atau butir untuk ekstasi)."""
    if substance == "ekstasi":
        if role == "pemakai":
            return round(random.uniform(1, 4), 1)     # 1–4 butir
        elif role in ("pemilik",):
            return round(random.uniform(3, 20), 1)
        else:  # kurir/penjual/pengendali
            return round(random.uniform(10, 200), 1)
    else:
        # gram
        if substance == "sabu":
            if role == "pemakai":
                return round(random.uniform(0.05, 0.5), 2)
            elif role in ("pemilik",):
                return round(random.uniform(0.2, 5.0), 2)
            else:
                return round(random.uniform(1.0, 200.0), 2)
        if substance == "ganja":
            if role == "pemakai":
                return round(random.uniform(0.5, 5.0), 2)
            elif role in ("pemilik",):
                return round(random.uniform(5.0, 200.0), 2)
            else:
                return round(random.uniform(20.0, 2000.0), 2)
        if substance == "heroin":
            if role == "pemakai":
                return round(random.uniform(0.05, 0.5), 2)
            elif role in ("pemilik",):
                return round(random.uniform(0.2, 3.0), 2)
            else:
                return round(random.uniform(1.0, 100.0), 2)
    return 1.0

def _pasal_role(role: str) -> Tuple[str, str]:
    """Mapping role → (pasal, kategori) tentang ancaman pidana narkotika (sangat disederhanakan)."""
    if role == "pemakai":
        return "127", "penyalahguna_rehabilitasi"
    if role == "pemilik":
        return "112", "narkotika_possession"
    # kurir/penjual/pengendali
    return "114", "narkotika_peredaran"

def __sentencing(pasal: str, substance: str, bb: float, unit: str, role: str) -> Dict[str, str]:
    """Heuristik rekomendasi hukuman tentang ancaman pidana narkotika (sangat disederhanakan) + alasan singkat."""
    if pasal == "127":  # pemakai
        # Indikasi rehabilitasi bila kecil
        if (substance == "ekstasi" and bb <= 3) or (substance != "ekstasi" and bb <= 0.5 if substance in ("sabu","heroin") else bb <= 5):
            return {
                "rekomendasi": "Pertimbangkan rehabilitasi",
                "rentang_": "hingga 4 tahun (maksimal) – ",
                "alasan": "BB kecil, role pemakai, asesmen diperlukan"
            }
        return {
            "rekomendasi": "Pidana pendek atau rehabilitasi (asesmen)",
            "rentang_": "hingga 4 tahun (maksimal) – ",
            "alasan": "Role pemakai; nilai BB moderat"
        }
    if pasal == "112":  # possession
        if (substance == "ekstasi" and bb < 10) or (substance != "ekstasi" and bb < 1.0):
            band = "4–8 tahun"
        elif (substance == "ekstasi" and bb < 50) or (substance != "ekstasi" and bb < 10.0):
            band = "6–12 tahun"
        else:
            band = "8–12 tahun"
        return {
            "rekomendasi": "Pidana penjara tentang ancaman pidana narkotika (sangat disederhanakan)",
            "rentang_": f"{band} – ",
            "alasan": f"Role possession; {bb} {unit} {substance}"
        }
    # pasal 114 peredaran
    if (substance == "ekstasi" and bb < 50) or (substance != "ekstasi" and bb < 5.0):
        band = "5–10 tahun"
    elif (substance == "ekstasi" and bb < 200) or (substance != "ekstasi" and bb < 50.0):
        band = "8–15 tahun"
    else:
        band = "12–20 tahun"
    return {
        "rekomendasi": "Pidana penjara tentang ancaman pidana narkotika (sangat disederhanakan)",
        "rentang_": f"{band} – ",
        "alasan": f"Role {role} (peredaran); {bb} {unit} {substance}"
    }

def _title_for_case(docket: str, role: str, substance: str, bb: float, unit: str) -> str:
    label = f"{role}, {substance} {bb}{unit[0]}"
    return f" Putusan PN – {docket} ({label})"

def generate__narkoba_cases(n: int = 200, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    docs = []
    for _ in range(n):
        court = random.choice(COURTS)
        year = random.randint(2021, 2025)
        nomor = f"{random.randint(1, 999):03d}/Pid.Sus/{year}/{court}"
        role = random.choices(ROLES, weights=[0.35, 0.25, 0.25, 0.10, 0.05], k=1)[0]
        substance = random.choices(SUBSTANCES, weights=[0.45, 0.35, 0.10, 0.10], k=1)[0]
        unit = _bb_unit(substance)
        bb = _rand_bb(substance, role)

        pasal, kategori = _pasal_role(role)
        reco = __sentencing(pasal, substance, bb, unit, role)

        title = _title_for_case(nomor, role, substance, bb, unit)
        konten = f"""
{title}

Fakta singkat:
- Peran terdakwa: {role}
- Zat: {substance} (Gol. I – disederhanakan)
- Barang bukti: {bb} {unit}
- Lokasi pengadilan: {court}
- Tahun perkara: {year}

Dakwaan/Pasal:
- Utama: Pasal {pasal} (, ringkas)
- Alternatif (bila relevan secara ): lihat Ringkasan/Template

Pertimbangan (, ringkas):
- Peran: {role}; barang bukti tercatat {bb} {unit}; tidak/ada indikasi peredaran tergantung peran.
- Keadaan meringankan/memberatkan tentang ancaman pidana narkotika (sangat disederhanakan): kooperatif, riwayat, dukungan keluarga, jaringan.

Amar tentang ancaman pidana narkotika (sangat disederhanakan):
- Rekomendasi sistem tentang ancaman pidana narkotika (sangat disederhanakan): {reco['rekomendasi']}
- Rentang ancaman tentang ancaman pidana narkotika (sangat disederhanakan): {reco['rentang_']}
- Alasan singkat: {reco['alasan']}

""".strip()

        docs.append({
            "title": title,
            "content": konten,
            "metadata": {
                "source": "putusan_pn_",
                "nomor_putusan": nomor,
                "tahun": str(year),
                "pengadilan": court,
                "jenis_perkara": "narkotika",
                "kategori": kategori,
                "role": role,
                "pasal": pasal,
                "zat": substance,
                "bb_nilai": bb,
                "bb_unit": unit,
                "rekomendasi_": reco["rekomendasi"],
                "rentang_": reco["rentang_"],
                "alasan_": reco["alasan"],
                "tags": ["", "narkotika", role, substance, f"pasal_{pasal}"],
            }
        })
    return docs


# ============== Router query WA-like ==============
def route_query(user_text: str) -> str:
    t = (user_text or "").lower()
    if "kasus" in t or "cerita" in t or "bla" in t:
        return "klasifikasi kasus narkotika pasal terkait jenis perbuatan pemakai possession peredaran definisi rehabilitasi router intent"
    if "berapa" in t or "hukuman" in t or "tahun" in t or "ancaman" in t:
        return "berapa tahun hukuman narkoba ancaman pidana pasal 112 114 127 rentang hukuman rehabilitasi pemakai possession peredaran"
    if "saran" in t or "gimana" in t or "bagaimana" in t or "rekomendasi" in t:
        return "saran strategi rehabilitasi assessment tim terpadu komunikasi klien template analisis cepat"
    return t


# ============== Main pipeline ==============
async def main():
    parser = argparse.ArgumentParser(description="Generate & load  narkoba cases into ParentChildRetriever.")
    parser.add_argument("--n", type=int, default=250, help="Jumlah kasus sintetis yang akan dibuat")
    parser.add_argument("--seed", type=int, default=42, help="Seed random untuk reproduksibilitas")
    parser.add_argument("--collection", type=str, default="putusan_child_chunks", help="Nama koleksi vectorstore")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K hasil retrieval saat tes")
    parser.add_argument("--include_fixed", action="store_true", help="Ikut masukkan dokumen ringkasan/faq/pedoman")
    args = parser.parse_args()

    try:
        print(f"🚀 Start generator  narkoba: N={args.n}, seed={args.seed}")
        random.seed(args.seed)

        # Init services & chains
        print("🔧 Initializing services/chains...")
        _ = get_llm_service()  # parity dengan arsitektur kamu
        rag_chains = create_rag_chains()
        embeddings = rag_chains.embeddings
        vector_store = rag_chains.vector_store

        print("🏗️ Initializing ParentChildRetriever...")
        retriever = ParentChildRetriever(
            vector_store=vector_store,
            embeddings_model=embeddings,
            collection_name=args.collection,
            child_chunk_size=400,
            child_chunk_overlap=50,
            parent_chunk_size=2000,
            parent_chunk_overlap=200,
        )

        # Siapkan dokumen
        print("📝 Generating synthetic cases...")
        generated = generate__narkoba_cases(n=args.n, seed=args.seed)
        docs: List[Document] = []
        now_iso = datetime.now().isoformat()

        # Tambahkan fixed (opsional)
        payload_docs = []
        if args.include_fixed:
            payload_docs.extend(FIXED_DOCS)
        payload_docs.extend(generated)

        for i, d in enumerate(payload_docs):
            meta = {
                **d["metadata"],
                "title": d["title"],
                "document_id": f"narkoba_doc_{i+1:06d}",
                "created_at": now_iso,
                "source_type": "_data",
                "language": "id",
                "domain": "narkotika",
            }
            docs.append(Document(page_content=d["content"].strip(), metadata=meta))
            if (i + 1) % 100 == 0 or i < 10:
                print(f"📄 Prepared doc {i+1}: {d['title'][:70]}...")

        print(f"\n💾 Adding {len(docs)} documents to collection '{args.collection}' ...")
        result = retriever.add_documents(docs)

        print("\n✅ Successfully added  documents!")
        try:
            print(f"   - Documents added: {result.get('added_documents')}")
            print(f"   - Total documents: {result.get('total_documents')}")
            print(f"   - Collection name: {result.get('collection_name')}")
        except Exception:
            print("   (Result shape differs) Raw:", result)

        # Tes retrieval dengan WA-like queries + skenario spesifik
        print("\n🔍 Testing retrieval ...")
        test_messages = [
            "Saya punya kasus bla2",
            "Berapa tahun hukumannya?",
            "Saran dari anda gimana",
            "kurir sabu 3 gram",
            "pemakai ganja 1 gram bisa rehabilitasi?",
        ]
        for msg in test_messages:
            q = route_query(msg)
            print(f"\n💬 Input: {msg}")
            print(f"🔎 Routed query: {q}")
            req = {"query": q, "top_k": args.top_k, "min_score": 0.0}
            try:
                resp = retriever.retrieve(req)
                docs_found = getattr(resp, "documents", []) or []
                total_found = getattr(resp, "total_found", None)
                print(f"   📋 Found: {total_found if total_found is not None else len(docs_found)}")
                for i, d in enumerate(docs_found[:args.top_k], 1):
                    meta = getattr(d, "metadata", {}) or {}
                    title = meta.get("title", "No title")
                    score = getattr(d, "score", None)
                    score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                    role = meta.get("role")
                    pasal = meta.get("pasal")
                    zat = meta.get("zat")
                    bb_val = meta.get("bb_nilai")
                    bb_unit = meta.get("bb_unit")
                    print(f"      {i}. {title[:90]} (score: {score_str}) | role={role}, pasal={pasal}, bb={bb_val}{bb_unit and ' '+bb_unit or ''}, zat={zat}")
            except Exception as e:
                print(f"   ❌ Query failed: {e}")

        # Strategy info
        print("\n📋 Retriever Strategy Info:")
        try:
            info = retriever.get_strategy_info()
            for k, v in info.items():
                print(f"   - {k}: {v}")
        except Exception as e:
            print(f"   (could not get strategy info) {e}")

        print("\n🎉 Generation + ingestion completed.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    asyncio.run(main())
