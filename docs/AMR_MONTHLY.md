# Data AMR bulanan

Simpan file bulanan berformat sama (.xlsx, .xls, atau .csv) di
`datasets/amr/monthly/`. File asli tidak diubah. Setiap sheet Excel harus
berisi tabel pengukuran dengan kolom yang sama. File Excel sementara `~$`
diabaikan. Kode pelanggan dibaca sebagai teks; nol yang sudah hilang di file
sumber tidak dapat dipulihkan.

Loader menggabungkan file, mengurutkan pelanggan/tanggal, dan menghapus
record identik. Record berbeda pada pelanggan/timestamp yang sama ditolak
beserta contoh sumbernya. READ_DATE harus tanggal yang dapat diparse
(disarankan YYYY-MM-DD atau tanggal Excel); timestamp asli dipertahankan sampai agregasi harian.
Tidak ada interpolasi hari kosong. Window tujuh hari boleh melintasi bulan,
tetapi tidak pelanggan atau jeda tanggal. Split tetap berdasarkan pelanggan,
dan normalisasi validasi/test memakai statistik training.

Setelah file tersedia, jalankan dari root proyek:

```bash
source carla-env-py311/bin/activate
python carla_pretext.py --config_env configs/env_config.yaml --config_exp configs/amr_monthly_config.yaml
python carla_classification.py --config_env configs/env_config.yaml --config_exp configs/amr_monthly_config.yaml
python carla_evaluate.py --config_env configs/env_config.yaml --config_exp configs/amr_monthly_config.yaml
```

Artefak folder `monthly` disimpan terpisah dari dataset satu-file sebelumnya.
Jangan memakai checkpoint lama setelah mengubah data/window; gunakan root_dir
baru dalam config environment untuk eksperimen baru. Konfigurasi dataset lama
tetap memakai window 1.

Nearest/furthest neighbors sekarang dihitung per sampel, tanpa sampel itu
sendiri, dengan k dibatasi n-1. Hanya isi repository yang sudah terisi dipakai.
Jalankan ulang pretext sebelum klasifikasi agar indeks tetangga lama diganti.
Perhitungan eksak memakai blok untuk membatasi memori sementara, tetapi waktu
komputasinya tetap kuadratik terhadap jumlah window.

Tanpa kolom label, loader masih memakai aturan heuristik lama; hasil evaluasi
tersebut mengukur kesesuaian dengan aturan, bukan label anomali terverifikasi.

Uji tanpa file bulanan asli:

```bash
python -m unittest discover -s tests -v
```

## Pembacaan 1–3 kali sehari

Konfigurasi bulanan mengaktifkan `daily_aggregation: true`. Setiap pelanggan
memiliki ringkasan harian mean/min/max untuk 31 fitur pengukuran dan turunan,
ditambah reading_count (94 input LSTM). Fitur energi kumulatif tambahan tidak
ikut dijumlahkan. Ringkasan ini berbobot per pembacaan, bukan rata-rata waktu.
Min/max merupakan ekstrem yang teramati saja. Dengan satu pembacaan,
mean=min=max; reading_count membantu membedakannya.

Duplikasi identik dihapus sebelum agregasi. Pembacaan berbeda di timestamp
berbeda pada hari sama diterima. Jika sumber hanya menyimpan tanggal tanpa
jam, record berbeda pada tanggal itu tetap ditolak: waktu pembacaan perlu
dipertahankan agar duplikasi dan pembacaan sah dapat dibedakan.

Label harian mengambil kode tertinggi seperti label window: Theft (2)
mendahului Defect (1), lalu Normal (0). Ini penyederhanaan single-label jika
kedua anomali terjadi bersamaan. Label/ID/tanggal bukan input model.

Hari kosong tidak diberi nilai nol atau interpolasi. Window dengan jeda
hari dilewati dan jumlahnya dicetak. Tidak diperlukan fitur missing-mask
karena hanya hari teramati masuk model pada kebijakan ini. Evaluasi dilakukan
pada window lengkap, sehingga tidak mewakili periode tanpa pembacaan.

Statistik normalisasi dihitung setelah agregasi hanya dari pelanggan training.
Artefak baru memakai suffix `_daily_stats`, terpisah dari model 31 fitur lama.
Metode ini adalah baseline konservatif; pilih kebijakan imputasi selanjutnya
berdasarkan cakupan window dan hasil validasi data nyata, bukan skor test.
