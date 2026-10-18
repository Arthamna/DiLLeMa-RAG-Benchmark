import pandas as pd

# Data for the CSV
data = [
    {
        "question": "Apa saja jenis program magang yang diselenggarakan oleh Kampus Merdeka di ITS?",
        "context": "Jenis program magang meliputi MBKM dan Non-MBKM. MBKM adalah program dari Kementerian Pendidikan, Kebudayaan, Riset, dan Teknologi dalam bentuk pembelajaran di luar kampus selama 2 semester yang bisa/wajib dikonversikan ke SKS mata kuliah. Non-MBKM adalah program magang yang bisa dikonversikan ke SKEM atau hanya untuk menambah pengalaman mahasiswa."
    },
    {
        "question": "Apa itu Program Magang Flagship di ITS?",
        "context": "Program Magang Flagship (Kampus Merdeka) meliputi MSIB (Magang dan Studi Independen Bersertifikat), Bangkit (by Google, GoTo, and Traveloka), dan Gerilya (by Kementerian ESDM)."
    },
    {
        "question": "Apa saja persyaratan untuk mengikuti program magang MBKM?",
        "context": "1. Mahasiswa aktif ITS tingkat Sarjana dan Diploma\n2. Minimal mahasiswa sudah lulus 90 SKS atau sudah lulus semester 5\n3. Mahasiswa tidak diperkenankan mengambil Program Kegiatan Magang (BKP) MBKM yang sama lebih dari 1 kali\n4. Mahasiswa wajib melakukan konversi alih kredit ke SKS MK\n5. Mahasiswa/Departemen wajib membuat/memiliki surat PKS\n6. Pelaksanaan Magang sebelum minggu ke-3 perkuliahan di ITS\n7. Mahasiswa tidak diperkenankan mengambil MK, atau hanya diperkenankan mengambil maks. 3 MK (setara 10 SKS)\n8. Pelaksanaan Magang selama 3 – 6 bulan."
    },
    {
        "question": "Apa saja persyaratan untuk mengikuti program magang Non-MBKM?",
        "context": "1. Mahasiswa aktif ITS tingkat Sarjana dan Diploma\n2. Minimal mahasiswa sudah lulus semester 3\n3. Mahasiswa dapat melakukan konversi alih kredit ke SKEM\n4. Tidak wajib membuat/memiliki surat PKS\n5. Pelaksanaan Magang di luar jam perkuliahan aktif (selama masa libur perkuliahan atau mahasiswa dapat mengajukan izin cuti perkuliahan)\n6. Pelaksanaan Magang selama 1 – 6 bulan."
    },
    {
        "question": "Bagaimana skema pelaksanaan Program Magang Flagship (MSIB)?",
        "context": "1. Mahasiswa membuat akun di https://kampusmerdeka.kemdikbud.go.id/\n2. Mahasiswa memilih program dan mitra yang akan diikuti\n3. Mahasiswa berkonsultasi ke Departemen serta mengajukan Surat Rekomendasi (SR)\n4. Departemen cek relevansi kegiatan/program mitra serta rencana konversi SKS\n5. Mahasiswa mengajukan SPTJM\n6. PK2 melakukan pengecekan Track Record Magang Mahasiswa\n7. SPTJM terbit\n8. Mahasiswa mendaftar program MSIB di website Kampus Merdeka dengan upload dokumen kelengkapan\n..."
    },
    {
        "question": "Bagaimana skema pelaksanaan Program Magang Kerjasama di ITS?",
        "context": "1. Mitra mengajukan permintaan mahasiswa magang ke PK2\n2. PK2 melakukan review dan persetujuan permohonan\n3. Mitra mengirim surat permintaan magang sesuai kualifikasi yang ditujukan ke Kasubdit. PK2 ITS\n4. PK2 mengirimkan draft PKS Magang Kerjasama kepada Mitra\n5. PK2 dan Mitra berkoordinasi untuk pengurusan PKS\n6. Mitra mengirim poster lowongan magang (jika ada)\n7. PK2 membuat form pendaftaran magang\n8. PK2 mempublikasikan ke sosmed dan menyebarkan informasi ke LO MBKM dan Kepala Departemen ... "
    },
    {
        "question": "Bagaimana skema pelaksanaan Program Magang Mandiri di ITS?",
        "context": "1. Mahasiswa mengurus surat rekomendasi departemen untuk pengajuan magang ke Mitra\n2. Mahasiswa mengajukan permohonan magang ke Mitra\n3. Mitra melakukan seleksi magang\n4. Mahasiswa menerima hasil seleksi magang dari Mitra\n5. Apabila tidak diterima, maka mahasiswa harus mengulang sejak langkah awal\n6. Apabila diterima, Mahasiswa melaporkan ke Departemen\n7. Mahasiswa/Departemen melakukan koordinasi dengan PK2 untuk pengurusan PKS dengan menyertakan Proposal Magang, Surat Rekomendasi Departemen dan Surat Penerimaan Magang\n8. Proses pengurusan PKS (Dapat dilakukan bersamaan dengan pelaksanaan Magang)\n9. Apabila PKS tidak disetujui ..."
    },
    {
        "question": "Apa statistik peserta MSIB Batch 4 dan 5?",
        "context": "MSIB Batch 4: Total Mahasiswa diterima: 664 mahasiswa; Program Magang: 228 mahasiswa; Program Studi Independen: 436 mahasiswa; Jumlah Mitra Industri: 108 Mitra\nMSIB Batch 5: Total Mahasiswa diterima: 488 mahasiswa; Program Magang: 257 mahasiswa; Program Studi Independen: 231 mahasiswa; Jumlah Mitra Industri: 121 Mitra"
    },
    {
        "question": "Apa saja tugas dari Penanggung Jawab Fakultas atau Prodi?",
        "context": "Penanggung Jawab Fakultas atau Prodi bertugas untuk:\n1. Melakukan verifikasi kebenaran data dan dokumen mahasiswa yang akan mendaftar Program flagship MBKM\n2. Memberikan rekomendasi kepada mahasiswa untuk mengikuti Program flagship MBKM\n3. Melakukan koordinasi dengan mahasiswa apabila terdapat data atau dokumen yang perlu diperbaiki."
    },
    {
        "question": "Bagaimana alur verifikasi dan persetujuan data mahasiswa oleh Penanggung Jawab Fakultas atau Prodi?",
        "context": "1. Penanggung Jawab Fakultas/Prodi selanjutnya bisa melakukan pengecekan kesesuaian data akademik pada Mahasiswa yang bersangkutan\n2. Jika data sudah sesuai, dan Anda sebagai Penanggung Jawab Fakultas/Prodi mengizinkan mahasiswa untuk mengikuti program Kampus Merdeka maka Anda bisa klik “Ya, Beri Izin”\n3. Untuk data mahasiswa yang sudah pernah diverifikasi, Anda akan menemukan notifikasi seperti yang ada pada gambar di samping\n4. Setelah itu klik “Selanjutnya”\n..."
    },
    {
        "question": "Bagaimana proses digitalisasi Surat Rekomendasi dan SPTJM?",
        "context": "1. Mahasiswa mengisikan seluruh data dan pernyataan mahasiswa langsung di dalam platform MBKM\n2. Proses verifikasi data mahasiswa dilakukan oleh Perguruan Tinggi pada level Penanggung Jawab Program Studi/Fakultas\n3. Rekomendasi Perguruan Tinggi diberikan melalui platform oleh Pimpinan PT\n4. Terdapat dua proses yang perlu dilakukan oleh Perguruan Tinggi:\n    a. Pengunggahan Surat Tugas dari PT\n    b. Submit email program studi atau pihak yang akan melakukan verifikasi mahasiswa\n..."
    },
    {
        "question": "Bagaimana cara penanggung jawab fakultas/prodi menerima email notifikasi penugasan verifikasi data akademik mahasiswa?",
        "context": "Penanggung Jawab Fakultas/Prodi akan menerima email notifikasi penugasan verifikasi data akademik mahasiswa untuk prodi tersebut untuk program Kampus Merdeka. Klik 'Verifikasi Sekarang'. Contoh email notifikasi penugasan jika belum ada mahasiswa yang terdaftar Mohon cek folder Spam jika email tidak muncul di folder Kotak Masuk."
    }
]

# Create dataframe
df = pd.DataFrame(data)

# Save to CSV
filepath = "./questions_contexts.csv"
df.to_csv(filepath, index=False)

filepath