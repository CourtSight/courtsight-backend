#!/usr/bin/env python3
"""
Script to add dummy legal documents to ParentChildRetriever for testing.
Creates 10 sample Indonesian legal documents and adds them to the retrieval system.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List

# Add the app directory to Python path
sys.path.append('/code')

from langchain_core.documents import Document
from app.services.retrieval.parent_child import ParentChildRetriever
from app.services.llm_service import get_llm_service
from app.services.rag.chains import create_rag_chains
from app.core.config import settings

# Sample Indonesian legal documents
DUMMY_DOCUMENTS = [
    {
        "title": "Putusan Mahkamah Agung No. 123/Pid/2023",
        "content": """
        PUTUSAN MAHKAMAH AGUNG REPUBLIK INDONESIA
        No. 123/Pid/2023
        
        DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA
        
        Mahkamah Agung yang memeriksa perkara pidana dalam tingkat kasasi telah menjatuhkan putusan sebagai berikut dalam perkara:
        
        TERDAKWA: Ahmad Budi Santoso
        DAKWAAN: Pasal 362 KUHP tentang pencurian
        
        PERTIMBANGAN HUKUM:
        1. Bahwa berdasarkan fakta-fakta hukum yang terungkap di persidangan, terdakwa terbukti secara sah dan meyakinkan bersalah melakukan tindak pidana pencurian.
        2. Bahwa unsur-unsur Pasal 362 KUHP telah terpenuhi seluruhnya.
        3. Bahwa terdakwa tidak memiliki alasan pembenar atau alasan pemaaf.
        
        AMAR PUTUSAN:
        1. Menyatakan terdakwa Ahmad Budi Santoso terbukti secara sah dan meyakinkan bersalah melakukan tindak pidana pencurian.
        2. Menjatuhkan pidana penjara selama 1 (satu) tahun.
        
        Demikian diputuskan dalam rapat permusyawaratan Mahkamah Agung pada hari Senin tanggal 15 Januari 2023.
        """,
        "metadata": {
            "source": "putusan_ma",
            "nomor_putusan": "123/Pid/2023",
            "tanggal": "2023-01-15",
            "jenis_perkara": "pidana",
            "pasal": "362 KUHP",
            "kategori": "pencurian"
        }
    },
    {
        "title": "Undang-Undang No. 8 Tahun 1981 tentang KUHAP",
        "content": """
        UNDANG-UNDANG REPUBLIK INDONESIA
        NOMOR 8 TAHUN 1981
        TENTANG KITAB UNDANG-UNDANG HUKUM ACARA PIDANA
        
        BAB I
        KETENTUAN UMUM
        
        Pasal 1
        Dalam undang-undang ini yang dimaksud dengan:
        1. Penyidikan adalah serangkaian tindakan penyidik dalam hal dan menurut cara yang diatur dalam undang-undang ini untuk mencari serta mengumpulkan bukti yang dengan bukti itu membuat terang tentang tindak pidana yang terjadi dan guna menemukan tersangkanya.
        
        2. Penyidik adalah pejabat polisi negara Republik Indonesia atau pejabat pegawai negeri sipil tertentu yang diberi wewenang khusus oleh undang-undang untuk melakukan penyidikan.
        
        3. Penuntutan adalah tindakan penuntut umum untuk melimpahkan perkara pidana ke pengadilan negeri yang berwenang dalam hal dan menurut cara yang diatur dalam undang-undang ini dengan permintaan supaya diperiksa dan diputus oleh hakim di sidang pengadilan.
        
        4. Penuntut umum adalah jaksa yang diberi wewenang oleh undang-undang ini untuk melakukan penuntutan dan melaksanakan penetapan hakim.
        """,
        "metadata": {
            "source": "undang_undang",
            "nomor": "8/1981",
            "tahun": "1981",
            "tentang": "KUHAP",
            "kategori": "hukum_acara_pidana"
        }
    },
    {
        "title": "Putusan Mahkamah Konstitusi No. 46/PUU-VIII/2010",
        "content": """
        PUTUSAN MAHKAMAH KONSTITUSI REPUBLIK INDONESIA
        No. 46/PUU-VIII/2010
        
        DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA
        
        Mahkamah Konstitusi Republik Indonesia yang memeriksa, mengadili, dan memutus perkara konstitusi pada tingkat pertama dan terakhir, menjatuhkan putusan dalam perkara Pengujian Undang-Undang Nomor 1 Tahun 1974 tentang Perkawinan terhadap Undang-Undang Dasar Negara Republik Indonesia Tahun 1945.
        
        PEMOHON: Machica Mochtar, dkk.
        
        POKOK PERMOHONAN:
        Pengujian konstitusionalitas Pasal 2 ayat (2) dan Pasal 43 ayat (1) Undang-Undang Nomor 1 Tahun 1974 tentang Perkawinan.
        
        PERTIMBANGAN HUKUM:
        1. Bahwa hak anak yang dilahirkan dari perkawinan yang tidak dicatatkan sama dengan hak anak yang dilahirkan dari perkawinan yang dicatatkan.
        2. Bahwa negara wajib memberikan perlindungan hukum yang sama terhadap semua anak tanpa diskriminasi.
        
        AMAR PUTUSAN:
        Pasal 43 ayat (1) Undang-Undang Nomor 1 Tahun 1974 tentang Perkawinan bertentangan dengan Undang-Undang Dasar Negara Republik Indonesia Tahun 1945 sepanjang dimaknai menghilangkan hubungan perdata dengan ayah biologisnya.
        """,
        "metadata": {
            "source": "putusan_mk",
            "nomor_putusan": "46/PUU-VIII/2010",
            "tahun": "2010",
            "jenis": "judicial_review",
            "kategori": "perkawinan"
        }
    },
    {
        "title": "Peraturan Pemerintah No. 24 Tahun 2018 tentang Pelayanan Perizinan Berusaha",
        "content": """
        PERATURAN PEMERINTAH REPUBLIK INDONESIA
        NOMOR 24 TAHUN 2018
        TENTANG PELAYANAN PERIZINAN BERUSAHA TERINTEGRASI SECARA ELEKTRONIK
        
        BAB I
        KETENTUAN UMUM
        
        Pasal 1
        Dalam Peraturan Pemerintah ini yang dimaksud dengan:
        1. Pelayanan Perizinan Berusaha Terintegrasi Secara Elektronik yang selanjutnya disebut Online Single Submission (OSS) adalah Perizinan Berusaha yang diterbitkan oleh Lembaga OSS untuk dan atas nama menteri, pimpinan lembaga, gubernur, atau bupati/walikota kepada Pelaku Usaha melalui sistem elektronik yang terintegrasi.
        
        2. Perizinan Berusaha adalah legalitas yang diberikan kepada Pelaku Usaha untuk memulai dan menjalankan usaha dan/atau kegiatannya.
        
        3. Lembaga OSS adalah lembaga yang menyelenggarakan Pelayanan Perizinan Berusaha Terintegrasi Secara Elektronik.
        
        BAB II
        PELAYANAN PERIZINAN BERUSAHA
        
        Pasal 2
        Pelayanan Perizinan Berusaha dilaksanakan melalui sistem OSS yang diselenggarakan oleh Lembaga OSS.
        """,
        "metadata": {
            "source": "peraturan_pemerintah",
            "nomor": "24/2018",
            "tahun": "2018",
            "tentang": "OSS",
            "kategori": "perizinan_usaha"
        }
    },
    {
        "title": "Putusan Pengadilan Tinggi Jakarta No. 567/Pdt/2022",
        "content": """
        PUTUSAN PENGADILAN TINGGI JAKARTA
        No. 567/Pdt/2022
        
        DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA
        
        Pengadilan Tinggi Jakarta yang memeriksa dan mengadili perkara perdata dalam tingkat banding, telah menjatuhkan putusan dalam perkara antara:
        
        PEMBANDING/TERGUGAT: PT. Maju Bersama
        TERBANDING/PENGGUGAT: CV. Sukses Mandiri
        
        POKOK PERKARA:
        Wanprestasi kontrak jual beli properti komersial senilai Rp 5.000.000.000,-
        
        PERTIMBANGAN HUKUM:
        1. Bahwa berdasarkan Pasal 1338 KUH Perdata, perjanjian yang dibuat secara sah berlaku sebagai undang-undang bagi para pihak.
        2. Bahwa terbukti pembanding telah melakukan wanprestasi dengan tidak menyerahkan objek jual beli sesuai waktu yang diperjanjikan.
        3. Bahwa kerugian yang dialami terbanding dapat dibuktikan dengan dokumen-dokumen yang sah.
        
        AMAR PUTUSAN:
        1. Menguatkan putusan Pengadilan Negeri Jakarta Pusat No. 123/Pdt.G/2021/PN.Jkt.Pst.
        2. Menghukum pembanding untuk membayar ganti rugi sebesar Rp 500.000.000,- kepada terbanding.
        """,
        "metadata": {
            "source": "putusan_pt",
            "nomor_putusan": "567/Pdt/2022",
            "tahun": "2022",
            "jenis_perkara": "perdata",
            "kategori": "wanprestasi"
        }
    },
    {
        "title": "Keputusan Menteri Hukum dan HAM No. 123/2023",
        "content": """
        KEPUTUSAN MENTERI HUKUM DAN HAM REPUBLIK INDONESIA
        NOMOR 123 TAHUN 2023
        TENTANG PEDOMAN PELAKSANAAN PEMBERIAN GRASI
        
        MENIMBANG:
        a. bahwa berdasarkan Pasal 14 ayat (1) Undang-Undang Dasar Negara Republik Indonesia Tahun 1945, Presiden memberi grasi dan rehabilitasi dengan memperhatikan pertimbangan Mahkamah Agung;
        b. bahwa untuk memberikan kepastian hukum dalam pelaksanaan pemberian grasi diperlukan pedoman yang jelas;
        
        MENGINGAT:
        1. Undang-Undang Dasar Negara Republik Indonesia Tahun 1945;
        2. Undang-Undang Nomor 22 Tahun 2002 tentang Grasi;
        
        MEMUTUSKAN:
        
        Pasal 1
        Pedoman Pelaksanaan Pemberian Grasi sebagaimana dimaksud dalam Keputusan ini terdiri dari:
        a. prosedur pengajuan permohonan grasi;
        b. persyaratan administratif;
        c. mekanisme pertimbangan;
        d. tata cara penetapan.
        
        Pasal 2
        Permohonan grasi diajukan oleh terpidana atau keluarganya kepada Presiden melalui Menteri Hukum dan Hak Asasi Manusia.
        """,
        "metadata": {
            "source": "keputusan_menteri",
            "nomor": "123/2023",
            "tahun": "2023",
            "instansi": "kemenkumham",
            "kategori": "grasi"
        }
    },
    {
        "title": "Putusan Mahkamah Agung No. 789/K/Sip/2023",
        "content": """
        PUTUSAN MAHKAMAH AGUNG REPUBLIK INDONESIA
        No. 789/K/Sip/2023
        
        DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA
        
        Mahkamah Agung yang memeriksa perkara perdata khusus dalam tingkat kasasi, menjatuhkan putusan dalam perkara:
        
        PEMOHON KASASI/TERGUGAT: Bank Central Asia Tbk
        TERMOHON KASASI/PENGGUGAT: Sari Dewi Rahayu
        
        POKOK PERKARA:
        Sengketa kredit macet dan lelang jaminan fidusia kendaraan bermotor.
        
        PERTIMBANGAN HUKUM:
        1. Bahwa berdasarkan Undang-Undang Nomor 42 Tahun 1999 tentang Jaminan Fidusia, kreditur pemegang fidusia memiliki hak untuk melaksanakan eksekusi terhadap benda jaminan fidusia.
        2. Bahwa prosedur lelang eksekusi fidusia telah sesuai dengan ketentuan peraturan perundang-undangan.
        3. Bahwa debitur telah diberikan kesempatan yang cukup untuk melunasi kewajibannya.
        
        AMAR PUTUSAN:
        1. Menolak permohonan kasasi dari pemohon kasasi.
        2. Menghukum pemohon kasasi untuk membayar biaya perkara tingkat kasasi sebesar Rp 500.000,-.
        
        Demikian diputuskan dalam rapat permusyawaratan hakim pada hari Rabu tanggal 20 September 2023.
        """,
        "metadata": {
            "source": "putusan_ma",
            "nomor_putusan": "789/K/Sip/2023",
            "tahun": "2023",
            "jenis_perkara": "perdata_khusus",
            "kategori": "fidusia"
        }
    },
    {
        "title": "Undang-Undang No. 19 Tahun 2016 tentang ITE",
        "content": """
        UNDANG-UNDANG REPUBLIK INDONESIA
        NOMOR 19 TAHUN 2016
        TENTANG PERUBAHAN ATAS UNDANG-UNDANG NOMOR 11 TAHUN 2008
        TENTANG INFORMASI DAN TRANSAKSI ELEKTRONIK
        
        BAB I
        KETENTUAN UMUM
        
        Pasal 1
        Dalam Undang-Undang ini yang dimaksud dengan:
        1. Informasi Elektronik adalah satu atau sekumpulan data elektronik, termasuk tetapi tidak terbatas pada tulisan, suara, gambar, peta, rancangan, foto, electronic data interchange (EDI), surat elektronik (electronic mail), telegram, teleks, telecopy atau sejenisnya, huruf, tanda, angka, Kode Akses, simbol, atau perforasi yang telah diolah yang memiliki arti atau dapat dipahami oleh orang yang mampu memahaminya.
        
        2. Transaksi Elektronik adalah perbuatan hukum yang dilakukan dengan menggunakan Komputer, jaringan Komputer, dan/atau media elektronik lainnya.
        
        BAB VII
        PERBUATAN YANG DILARANG
        
        Pasal 27
        (1) Setiap Orang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan yang melanggar kesusilaan.
        
        (3) Setiap Orang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan dan/atau membuat dapat diaksesnya Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan penghinaan dan/atau pencemaran nama baik.
        """,
        "metadata": {
            "source": "undang_undang",
            "nomor": "19/2016",
            "tahun": "2016",
            "tentang": "ITE",
            "kategori": "teknologi_informasi"
        }
    },
    {
        "title": "Putusan Pengadilan Negeri Surabaya No. 345/Pid.Sus/2023",
        "content": """
        PUTUSAN PENGADILAN NEGERI SURABAYA
        No. 345/Pid.Sus/2023
        
        DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA
        
        Pengadilan Negeri Surabaya yang memeriksa dan mengadili perkara pidana khusus, menjatuhkan putusan dalam perkara:
        
        TERDAKWA: Rizki Pratama
        DAKWAAN: Pasal 45 ayat (2) Jo. Pasal 27 ayat (3) UU No. 19 Tahun 2016 tentang ITE
        
        FAKTA HUKUM:
        1. Bahwa terdakwa telah mengunggah video yang berisi penghinaan terhadap korban di media sosial Instagram.
        2. Bahwa video tersebut telah ditonton oleh ratusan pengguna dan menimbulkan dampak psikologis bagi korban.
        3. Bahwa perbuatan terdakwa memenuhi unsur-unsur pencemaran nama baik melalui media elektronik.
        
        PERTIMBANGAN HUKUM:
        1. Bahwa dakwaan jaksa penuntut umum telah terbukti secara sah dan meyakinkan.
        2. Bahwa perbuatan terdakwa meresahkan masyarakat dan mencoreng martabat korban.
        3. Bahwa terdakwa menunjukkan penyesalan dan bersedia meminta maaf kepada korban.
        
        AMAR PUTUSAN:
        1. Menyatakan terdakwa Rizki Pratama terbukti secara sah dan meyakinkan bersalah melakukan tindak pidana pencemaran nama baik melalui media elektronik.
        2. Menjatuhkan pidana penjara selama 8 (delapan) bulan dengan masa percobaan 1 (satu) tahun.
        """,
        "metadata": {
            "source": "putusan_pn",
            "nomor_putusan": "345/Pid.Sus/2023",
            "tahun": "2023",
            "pengadilan": "PN Surabaya",
            "jenis_perkara": "pidana_khusus",
            "kategori": "cybercrime"
        }
    },
    {
        "title": "Peraturan Mahkamah Agung No. 2 Tahun 2023 tentang E-Court",
        "content": """
        PERATURAN MAHKAMAH AGUNG REPUBLIK INDONESIA
        NOMOR 2 TAHUN 2023
        TENTANG PERSIDANGAN ELEKTRONIK
        
        BAB I
        KETENTUAN UMUM
        
        Pasal 1
        Dalam Peraturan Mahkamah Agung ini yang dimaksud dengan:
        1. Persidangan Elektronik yang selanjutnya disebut e-Court adalah persidangan yang dilaksanakan secara elektronik dengan menggunakan sistem teknologi informasi.
        
        2. Sistem e-Court adalah sistem teknologi informasi yang digunakan untuk menyelenggarakan administrasi perkara dan persidangan secara elektronik.
        
        BAB II
        RUANG LINGKUP
        
        Pasal 2
        (1) e-Court dapat digunakan untuk semua jenis perkara di lingkungan peradilan umum, peradilan agama, peradilan militer, dan peradilan tata usaha negara.
        
        (2) Pelaksanaan e-Court dilakukan dengan tetap memperhatikan asas sederhana, cepat, dan biaya ringan.
        
        Pasal 3
        Persidangan melalui e-Court dapat dilaksanakan untuk:
        a. pemeriksaan saksi dan/atau ahli;
        b. pembacaan putusan;
        c. mediasi;
        d. persidangan penuh dalam kondisi tertentu.
        """,
        "metadata": {
            "source": "perma",
            "nomor": "2/2023",
            "tahun": "2023",
            "tentang": "e-court",
            "kategori": "administrasi_peradilan"
        }
    }
]


async def main():
    """Main function to add dummy documents to ParentChildRetriever."""
    try:
        print("🚀 Starting dummy document addition process...")
        print(f"📝 Preparing {len(DUMMY_DOCUMENTS)} dummy legal documents")
        
        # Initialize LLM service and RAG chains
        print("🔧 Initializing services...")
        llm_service = get_llm_service()
        rag_chains = create_rag_chains()
        
        # Get embeddings and vector store from rag_chains
        embeddings = rag_chains.embeddings
        vector_store = rag_chains.vector_store
        
        print("🏗️ Initializing ParentChildRetriever...")
        
        # Initialize ParentChildRetriever
        retriever = ParentChildRetriever(
            vector_store=vector_store,
            embeddings_model=embeddings,
            collection_name="putusan_child_chunks",
            child_chunk_size=400,
            child_chunk_overlap=50,
            parent_chunk_size=2000,
            parent_chunk_overlap=200
        )
        
        # Convert dummy documents to LangChain Document format
        documents: List[Document] = []
        
        for i, doc_data in enumerate(DUMMY_DOCUMENTS):
            doc = Document(
                page_content=doc_data["content"].strip(),
                metadata={
                    **doc_data["metadata"],
                    "title": doc_data["title"],
                    "document_id": f"dummy_doc_{i+1:02d}",
                    "created_at": datetime.now().isoformat(),
                    "source_type": "dummy_data",
                    "language": "id"  # Indonesian
                }
            )
            documents.append(doc)
            print(f"📄 Prepared document {i+1}: {doc_data['title'][:50]}...")
        
        print(f"\n💾 Adding {len(documents)} documents to ParentChildRetriever...")
        
        # Add documents to retriever
        result = retriever.add_documents(documents)
        
        print("\n✅ Successfully added dummy documents!")
        print(f"📊 Addition Results:")
        print(f"   - Documents added: {result['added_documents']}")
        print(f"   - Total documents: {result['total_documents']}")
        print(f"   - Collection name: {result['collection_name']}")
        
        # Test retrieval with a sample query
        print("\n🔍 Testing document retrieval...")
        test_queries = [
            "pencurian KUHP",
            "wanprestasi kontrak",
            "UU ITE pencemaran nama baik",
            "grasi Presiden"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Testing query: '{query}'")
            
            # Create a simple request object
            request = {
                'query': query,
                'top_k': 3,
                'min_score': 0.0
            }
            
            try:
                response = retriever.retrieve(request)
                print(f"   📋 Found {response.total_found} documents")
                
                for i, doc in enumerate(response.documents[:2], 1):  # Show first 2 results
                    print(f"      {i}. {doc.metadata.get('title', 'No title')[:60]}... (score: {doc.score:.3f})")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {str(e)}")
        
        # Get strategy info
        print(f"\n📋 Retriever Strategy Info:")
        strategy_info = retriever.get_strategy_info()
        for key, value in strategy_info.items():
            print(f"   - {key}: {value}")
        
        print("\n🎉 Dummy document addition completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during document addition: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())