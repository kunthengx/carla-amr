# anomaly-injection-v2-complete — AMR 3-Phase + CARLA

Paket lengkap anomaly injection untuk data AMR 3-fasa dan pipeline CARLA.

## Fitur
- `location_code` + `reading_time`
- agregasi 1–3 pembacaan/hari menjadi 1/hari (default median)
- 7 tipe anomaly injection
- label 0=normal, 1=defect, 2=theft
- physics-aware P/Q/S recomputation
- anomaly report + statistik JSON + plot
- export CARLA window `[N,T,F]` untuk window 7/30/50
- notebook Google Colab dan smoke test

## Menjalankan
```bash
pip install -r requirements.txt
python inject_anomaly_v2.py --input Data_Sample_Instant.xlsx --config anomaly_config.yaml --output-dir sample_output
```

Window 7 hari:
```bash
python inject_anomaly_v2.py --input Data_Sample_Instant.xlsx --window-size 7
```

## Integrasi CARLA
Jika `AMR3PhaseDataset` Anda sudah membuat window sendiri, gunakan `Data_AMR_Anomaly_v2.xlsx` langsung dan jangan lakukan windowing kedua. Pertahankan group split berdasarkan `location_code`, dan hitung normalisasi hanya dari train set.

## Catatan
Min-max adalah scaling, bukan metode untuk menyatukan 1–3 pembacaan harian. Paket ini memakai median harian sebagai default karena lebih robust terhadap pembacaan ekstrem.
