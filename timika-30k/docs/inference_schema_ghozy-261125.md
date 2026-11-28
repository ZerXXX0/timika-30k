# Skema Inferensi Model Timika-30K

## Inferensi yang dijalankan

| Nomor | Model | Bentuk citra masukan | Folder citra pada dataset |
|---|---|---|---|
| 1 | SegFormer | Citra bone | derived-data-bone_suppressed-resnet_bs/images |
| 2 | U-Net++ | Citra bone | derived-data-bone_suppressed-resnet_bs/images |
| 3 | U-Net++ | Citra contrast | derived-data-clahe/images |
| 4 | U-Net++ | Citra bone and contrast | derived-data-bone_suppressed_clahe-resnet_bs/images |

## Pengaturan inferensi

| Pengaturan | Nilai |
|---|---|
| Citra uji | 553 citra uji ChestX-Det, daftar id pada berkas daftar_citra_uji_chestxdet-261125.csv |
| Checkpoint | Epoch terbaik pada validation |
| Ukuran masukan | 512 × 512, satu kanal abu-abu |
| Ambang keluaran sigmoid | 0.5 |
| Komponen terhubung yang dihapus | Kurang dari 30 piksel |
| Presisi | fp16 |
| Kelas yang disimpan | atelectasis, infiltrate, pleural effusion, pneumothorax |

## Berkas yang dikirim

| Berkas | Isi |
|---|---|
| Mask prediksi | Satu berkas PNG biner 512 × 512 per citra dan per kelas, untuk setiap inferensi pada tabel |
| Tabel probabilitas | Satu berkas CSV per inferensi, satu baris per citra, dengan nilai tertinggi keluaran sigmoid pada citra tersebut untuk setiap kelas |
