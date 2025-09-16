#!/usr/bin/env python3
"""
Add detailed KUHP documents to ParentChildRetriever
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.retrieval import get_retrieval_service
from app.services.retrieval.base import RetrievalStrategy
from langchain_core.documents import Document

def create_detailed_kuhp_documents():
    """Create detailed KUHP documents with specific articles."""
    
    documents = [
        Document(
            page_content="""KITAB UNDANG-UNDANG HUKUM PIDANA (KUHP)
            
BAB XXII
PENCURIAN

Pasal 362

Barang siapa mengambil barang sesuatu, yang seluruhnya atau sebagian kepunyaan orang lain, dengan maksud untuk dimiliki secara melawan hukum, diancam karena pencurian dengan pidana penjara paling lama lima tahun atau pidana denda paling banyak sembilan ratus rupiah.

Penjelasan:
- Pencurian adalah tindak pidana mengambil barang milik orang lain
- Maksud untuk memiliki secara melawan hukum harus terbukti
- Sanksi: pidana penjara paling lama 5 tahun atau denda paling banyak 900 rupiah

Pasal 363

Diancam dengan pidana penjara paling lama tujuh tahun:
1. pencurian ternak;
2. pencurian pada waktu ada kebakaran, letusan, banjir, gempa bumi atau gempa laut, gunung meletus atau ada bahaya umum lainnya;
3. pencurian di waktu malam dalam sebuah rumah atau pekarangan tertutup yang ada rumahnya, yang dilakukan oleh orang yang ada di situ tidak diketahui atau tidak dikehendaki oleh yang berhak;
4. pencurian yang dilakukan oleh dua orang atau lebih dengan bersekutu;
5. pencurian yang untuk masuk ke tempat melakukan kejahatan, atau untuk sampai pada barang yang diambil, dilakukan dengan merusak, memotong atau memanjat, atau dengan memakai anak kunci palsu, perintah palsu atau pakaian jabatan palsu.

Pasal 364

Perbuatan yang diterangkan dalam pasal 362 dan pasal 363 ke-4 dan ke-5, apabila tidak dilakukan dalam sebuah rumah atau pekarangan tertutup yang ada rumahnya, dan harga barang yang dicuri tidak lebih dari dua puluh lima rupiah, diancam sebagai pencurian ringan dengan pidana penjara paling lama tiga bulan atau pidana denda paling banyak dua ratus lima puluh rupiah.""",
            metadata={
                "title": "KUHP Bab XXII - Pencurian (Pasal 362-364)",
                "kategori": "hukum_pidana",
                "nomor": "KUHP-362-364",
                "source": "KUHP",
                "document_type": "undang_undang",
                "jenis_dokumen": "kodifikasi_hukum",
                "bidang_hukum": "pidana"
            }
        ),
        
        Document(
            page_content="""KITAB UNDANG-UNDANG HUKUM PIDANA (KUHP)
            
BAB XXIII
PENADAHAN

Pasal 480

Diancam dengan pidana penjara paling lama empat tahun atau pidana denda paling banyak sembilan ratus rupiah:
1. barang siapa membeli, menyewa, menerima tukar, menerima gadai, menerima hibah, atau karena hendak mendapat untung, menjual, menyewakan, menukarkan, menggadaikan, mengangkut, menyimpan atau menyembunyikan sesuatu benda, yang diketahuinya atau sepatutnya harus diduganya bahwa benda itu diperoleh karena kejahatan;
2. barang siapa menguntungkan diri dari hasil sesuatu benda, yang diketahuinya atau sepatutnya harus diduganya bahwa benda itu diperoleh karena kejahatan.

Pasal 481

Pidana yang ditentukan dalam pasal yang lalu dapat ditambah sepertiga:
1. jika yang bersalah melakukan kejahatan tersebut dalam pasal yang lalu sebagai pencaharian;
2. jika yang bersalah adalah orang yang karena pekerjaannya atau jabatannya mempunyai tugas khusus untuk memberantas kejahatan.""",
            metadata={
                "title": "KUHP Bab XXIII - Penadahan (Pasal 480-481)",
                "kategori": "hukum_pidana",
                "nomor": "KUHP-480-481",
                "source": "KUHP",
                "document_type": "undang_undang",
                "jenis_dokumen": "kodifikasi_hukum",
                "bidang_hukum": "pidana"
            }
        ),
        
        Document(
            page_content="""KITAB UNDANG-UNDANG HUKUM PIDANA (KUHP)
            
BAB XV
KEJAHATAN TERHADAP KEAMANAN UMUM

Pasal 187

Barang siapa dengan sengaja menyebabkan kebakaran, ledakan atau banjir, diancam:
1. dengan pidana penjara paling lama dua belas tahun, jika karena perbuatan tersebut timbul bahaya umum bagi barang;
2. dengan pidana penjara paling lama lima belas tahun, jika karena perbuatan tersebut timbul bahaya bagi nyawa orang lain;
3. dengan pidana penjara seumur hidup atau pidana mati, jika karena perbuatan tersebut mengakibatkan orang lain mati.

Pasal 188

Barang siapa karena kealpaannya menyebabkan kebakaran, ledakan atau banjir, diancam dengan pidana penjara paling lama lima tahun atau pidana kurungan paling lama satu tahun atau pidana denda paling banyak empat ribu lima ratus rupiah, jika karena perbuatan tersebut:
1. timbul bahaya umum bagi barang; atau
2. timbul bahaya bagi nyawa orang lain; atau
3. mengakibatkan orang lain mati.""",
            metadata={
                "title": "KUHP Bab XV - Kejahatan Terhadap Keamanan Umum (Pasal 187-188)",
                "kategori": "hukum_pidana",
                "nomor": "KUHP-187-188",
                "source": "KUHP",
                "document_type": "undang_undang",
                "jenis_dokumen": "kodifikasi_hukum",
                "bidang_hukum": "pidana"
            }
        ),
        
        Document(
            page_content="""KITAB UNDANG-UNDANG HUKUM PIDANA (KUHP)
            
BAB XVI
KEJAHATAN TERHADAP KETERTIBAN UMUM

Pasal 212

Barang siapa dengan sengaja menolong seorang tahanan melarikan diri atau melepaskan diri dari tahanan yang sah, diancam dengan pidana penjara paling lama empat tahun atau pidana denda paling banyak empat ribu lima ratus rupiah.

Pasal 213

Jika kejahatan dalam pasal yang lalu dilakukan dengan kekerasan atau ancaman kekerasan terhadap orang atau dengan membongkar atau merusak tempat tahanan, maka yang bersalah diancam dengan pidana penjara paling lama enam tahun.

Pasal 214

Jika yang melakukan salah satu kejahatan yang diterangkan dalam kedua pasal yang lalu adalah penjaga tahanan atau orang lain yang ditugaskan dengan penjagaan atau pengangkutan tahanan, maka pidana dapat ditambah sepertiga.""",
            metadata={
                "title": "KUHP Bab XVI - Kejahatan Terhadap Ketertiban Umum (Pasal 212-214)",
                "kategori": "hukum_pidana",
                "nomor": "KUHP-212-214",
                "source": "KUHP",
                "document_type": "undang_undang",
                "jenis_dokumen": "kodifikasi_hukum",
                "bidang_hukum": "pidana"
            }
        ),
        
        Document(
            page_content="""KITAB UNDANG-UNDANG HUKUM PIDANA (KUHP)
            
BAB XVIII
KEJAHATAN TERHADAP KEMERDEKAAN ORANG

Pasal 333

Barang siapa dengan sengaja dan melawan hukum merampas kemerdekaan seseorang, atau meneruskan perampasan kemerdekaan yang demikian, diancam dengan pidana penjara paling lama delapan tahun.

Pasal 334

Jika perbuatan tersebut dalam pasal yang lalu mengakibatkan luka-luka berat, maka yang bersalah diancam dengan pidana penjara paling lama sembilan tahun.

Pasal 335

Jika mengakibatkan mati, diancam dengan pidana penjara paling lama dua belas tahun.

Pasal 368

Barang siapa dengan maksud untuk menguntungkan diri sendiri atau orang lain secara melawan hukum, memaksa seorang dengan kekerasan atau ancaman kekerasan untuk memberikan barang sesuatu, yang seluruhnya atau sebagian adalah kepunyaan orang itu atau orang lain, atau supaya membuat utang maupun menghapuskan piutang, diancam karena pemerasan dengan pidana penjara paling lama sembilan tahun.""",
            metadata={
                "title": "KUHP Bab XVIII - Kejahatan Terhadap Kemerdekaan Orang (Pasal 333-335, 368)",
                "kategori": "hukum_pidana",
                "nomor": "KUHP-333-368",
                "source": "KUHP",
                "document_type": "undang_undang",
                "jenis_dokumen": "kodifikasi_hukum",
                "bidang_hukum": "pidana"
            }
        )
    ]
    
    return documents

async def add_detailed_kuhp_documents():
    """Add detailed KUHP documents to the retrieval service."""
    
    try:
        print("=== Adding Detailed KUHP Documents ===")
        
        # Get retrieval service
        retrieval_service = get_retrieval_service()
        print(f"✓ Retrieved service: {type(retrieval_service).__name__}")
        
        # Create detailed documents
        documents = create_detailed_kuhp_documents()
        print(f"✓ Created {len(documents)} detailed KUHP documents")
        
        # Add documents using parent-child strategy
        result = retrieval_service.add_documents(
            documents=documents,
            strategy=RetrievalStrategy.PARENT_CHILD
        )
        
        print("✅ Addition Results:")
        print(f"   Documents added: {result.get('documents_added', 'N/A')}")
        print(f"   Total documents: {result.get('total_documents', 'N/A')}")
        print(f"   Collection name: {result.get('collection_name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding detailed KUHP documents: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(add_detailed_kuhp_documents())