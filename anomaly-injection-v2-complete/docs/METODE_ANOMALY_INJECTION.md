# Metode Anomaly Injection V2

Tipe anomali: current bypass theft, intermittent theft, phase loss, voltage imbalance, PF manipulation, neutral-current leakage, dan meter drift. Semua event dicatat ke `anomaly_report.csv`.

Anomaly injection dipakai sebagai augmentation/controlled benchmark, bukan pengganti data anomali nyata. Evaluasi tesis sebaiknya tetap menggunakan group split pelanggan untuk mencegah data leakage.
