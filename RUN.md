# Runbook

## Persyaratan RTX 5080

Runtime RETRO/PipeRAG memakai stack Blackwell yang diadaptasi dari
`feat/rankify-50`: PyTorch `2.13.0+cu132`, CUDA 13.2/cuDNN 9, ONNX Runtime GPU
1.27, dan Python 3.11–3.12. RTX 5080 memiliki compute capability 12.0
(`sm_120`). Driver NVIDIA minimal untuk keluarga CUDA 13.x adalah 580; driver
R595 atau lebih baru direkomendasikan untuk toolkit CUDA 13.2.

Verifikasi host sebelum build:

```bash
nvidia-smi
docker run --rm --gpus all \
  nvidia/cuda:13.2.1-base-ubuntu24.04 nvidia-smi
```

Sesudah `bash scripts/setup_ubuntu.sh`, script memeriksa versi runtime CUDA,
`sm_120`, provider ONNX CUDA, dan menjalankan alokasi tensor GPU. Perubahan
versi runtime ikut masuk fingerprint, sehingga artifact ONNX lama otomatis
dibangun ulang. Export pertama dapat memakai RAM/VRAM jauh lebih besar daripada
steady-state inference; pastikan tidak ada model lokal lain yang memakai GPU.

Jika instalasi host pernah berhenti dengan `ImportError: libcudart.so.13`, uji
environment dengan PyTorch diimpor sebelum ONNX Runtime:

```bash
uv run --no-sync python - <<'PY'
import torch
import onnxruntime as ort

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("ONNX Runtime:", ort.__version__)
print("Providers:", ort.get_available_providers())
PY
```

Wheel PyTorch membawa library CUDA 13 yang diperlukan; tidak perlu menyalin
`libcudart.so.13` ke `/usr/lib` atau membuat symlink manual. Urutan import di
atas penting ketika CUDA Toolkit 13 tidak dipasang secara global pada host.

## 1. Menyiapkan checkpoint RETRO di VPS

Weights tidak diunduh otomatis oleh repository dan direktori `models/` diabaikan
Git. RETRO dan PipeRAG harus memakai **file checkpoint yang sama**:

- checkpoint: `models/model.ckpt`
- konfigurasi: `models/retro.json`
- sumber resmi: RETRO commit
  [`6945cdd`](https://github.com/TobiasNorlund/retro/tree/6945cdd53f6472eaad78842c3b9a4a729f677acd)
- Google Drive file ID: `1R53kuW_6uWDCXamQy2AghgeseaIPsAcB`

Arsip upstream berukuran sekitar 6,90 GB. Siapkan sedikitnya 20 GB ruang kosong
untuk menampung arsip dan hasil ekstraksi secara bersamaan:

```bash
df -h .
mkdir -p models
```

### Mengunduh checkpoint

Google Drive memakai halaman konfirmasi untuk file besar, sehingga `curl -L`
langsung dapat menyimpan HTML alih-alih ZIP. Gunakan `gdown`; `uvx` menjalankannya
tanpa menambah dependency ke project:

```bash
uvx --from gdown gdown --continue \
  '1R53kuW_6uWDCXamQy2AghgeseaIPsAcB' \
  --output models/retro-upstream.zip
```

Opsi `--continue` memungkinkan command yang sama dijalankan lagi setelah koneksi
VPS terputus. Jangan lanjut ke ekstraksi sebelum ZIP lolos pemeriksaan:

```bash
unzip -t models/retro-upstream.zip
unzip -Z1 models/retro-upstream.zip | grep -E '(^|/)model\.ckpt$'
```

Ekstrak hanya checkpoint, terlepas dari apakah ZIP menyimpannya di root atau di
subdirektori:

```bash
set -euo pipefail

CHECKPOINT_MEMBER="$(
  unzip -Z1 models/retro-upstream.zip |
    awk -F/ '$NF == "model.ckpt" && !found { print; found = 1 }'
)"

if [ -z "$CHECKPOINT_MEMBER" ]; then
  echo 'model.ckpt tidak ditemukan di dalam ZIP' >&2
  exit 1
fi

unzip -p models/retro-upstream.zip "$CHECKPOINT_MEMBER" \
  > models/model.ckpt.partial
mv models/model.ckpt.partial models/model.ckpt
```

Jika `uvx` tidak tersedia, instal `gdown` ke virtual environment sementara:

```bash
python3 -m venv /tmp/retro-download-venv
/tmp/retro-download-venv/bin/pip install 'gdown>=5,<6'
/tmp/retro-download-venv/bin/gdown --continue \
  '1R53kuW_6uWDCXamQy2AghgeseaIPsAcB' \
  --output models/retro-upstream.zip
```

### Mengunduh konfigurasi dengan `curl`

Ambil `retro.json` langsung dari commit RETRO yang dikunci, bukan dari branch
`main` yang dapat berubah:

```bash
curl --fail --location --show-error \
  --retry 5 --retry-all-errors \
  'https://raw.githubusercontent.com/TobiasNorlund/retro/6945cdd53f6472eaad78842c3b9a4a729f677acd/data/model/retro.json' \
  --output models/retro.json.partial

printf '%s  %s\n' \
  '09e396f61b2e04d31d278d853504433a73beebac0bdcb5c7ee1626104cba3269' \
  'models/retro.json.partial' | sha256sum --check

mv models/retro.json.partial models/retro.json
python3 -m json.tool models/retro.json >/dev/null
```

Upstream tidak menerbitkan checksum checkpoint di README-nya. Setelah unduhan
pertama yang sukses, simpan hash lokal bersama hasil eksperimen agar file yang
sama dapat diverifikasi pada VPS atau run berikutnya:

```bash
test -s models/model.ckpt
test -s models/retro.json
sha256sum models/model.ckpt models/retro.json | tee models/SHA256SUMS.local
sha256sum --check models/SHA256SUMS.local
```

`models/SHA256SUMS.local` juga diabaikan Git karena berada di `models/`. Salin
hash-nya ke penyimpanan hasil/metadata yang persisten. Manifest benchmark akan
merekam SHA-256 checkpoint yang benar-benar dipakai.

Setelah hash tersimpan, arsip 6,90 GB boleh dihapus untuk menghemat ruang:

```bash
rm -- models/retro-upstream.zip
```

Verifikasi nama, permission, dan mount Compose sebelum membangun service:

```bash
ls -lh models/model.ckpt models/retro.json
docker compose config >/dev/null
```

Jika `RETRO_MODEL_DIR` di `.env` diarahkan ke lokasi lain, kedua file tersebut
harus ada di direktori itu. Di dalam container inference keduanya tetap terlihat
sebagai `/models/model.ckpt` dan `/models/retro.json`.

## 2. Setup service dan environment

1. Salin `.env.example` ke `.env`; isi model/kredensial DiLLeMa dan, bila perlu,
   ubah direktori volume Compose.
2. Jalankan `bash scripts/setup_ubuntu.sh`.
3. Jalankan `docker compose up --build -d`.
4. Jalankan `doctor` untuk preset target.
5. Jalankan stage `all` dan simpan output ITS/UniQA di direktori berbeda.

```bash
cp .env.example .env
mkdir -p artifacts/onnx artifacts/indexes artifacts/huggingface
bash scripts/setup_ubuntu.sh
docker compose up --build -d

set -a; source .env; set +a

uv run --no-sync python -m rag_benchmark doctor \
  --config configs/experiments/its.yaml

uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/its.yaml --stage all \
  --output artifacts/benchmarks/its

uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/uniqa.yaml --stage all \
  --output artifacts/benchmarks/uniqa
```

Saat `Configure` pertama, inference memuat checkpoint, mengekspor ONNX, lalu
memeriksa parity PyTorch–ONNX sebelum QA dimulai. Error `Add ... LeftShape
{1,1,1,64}, RightShape {1,1,65,65}` menandakan image lama yang masih memakai
KV-cache parity check sebelum perbaikan isolasi state. Rebuild dan recreate hanya
service inference, lalu pastikan simbol perbaikannya tersedia:

```bash
docker compose build inference
docker compose up -d --force-recreate inference
docker compose exec inference /opt/venv/bin/python -c \
  'from rag_benchmark.techniques.retro.onnx_runtime import clone_past_key_values; print("KV-cache fix loaded")'
```

Artifact ONNX dari Configure yang gagal tidak memiliki manifest valid dan akan
ditimpa otomatis pada percobaan berikutnya; tidak perlu menghapus checkpoint.

Selisih kernel FP32 yang sangat kecil dapat berbeda antar-generasi GPU. Parity
tetap mewajibkan token argmax yang sama, dengan batas absolut default `0.002`
untuk encoder dan decoder. Nilainya direkam dalam manifest dan dapat diatur di
`.env` melalui `RETRO_ONNX_ENCODER_ATOL` dan `RETRO_ONNX_DECODER_ATOL`. Jangan
menaikkan nilai tersebut hanya untuk melewati perbedaan token (`token_equal=false`).

Untuk hanya pasangan terkontrol:

```bash
uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/its.yaml --stage all \
  --techniques retro,piperag \
  --output artifacts/benchmarks/its-retro-piperag
```

### Batas memori GPU 16 GB

Run resmi memakai fase GPU yang tidak saling tumpang tindih:

1. service inference RETRO/PipeRAG di-unload;
2. DiLLeMa menyelesaikan retrieval dan reranking;
3. embedder/reranker dilepas dan cache CUDA dikosongkan;
4. generator lokal dimuat dan menyelesaikan generasi;
5. generator dilepas, kemudian RETRO/PipeRAG memakai checkpoint bersama;
6. service inference di-unload lagi setelah evaluasi.

Default konservatif untuk RTX 5080 16 GB adalah `EMBED_BATCH_SIZE=8`,
`RERANK_BATCH_SIZE=4`, dan
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Ubah di `.env`, lalu pastikan
variabel diekspor sebelum proses detach dibuat:

```bash
set -a; source .env; set +a
```

Inference RETRO membutuhkan model PyTorch untuk membuat cache awal, sedangkan
token berikutnya diproses oleh ONNX Runtime. Agar dua salinan bobot tidak menetap
di GPU yang sama, default Compose adalah:

```dotenv
RETRO_BOOTSTRAP_DEVICE=cpu
RETRO_ORT_ARENA_EXTEND_STRATEGY=kSameAsRequested
```

Pengaturan pertama menyimpan model bootstrap PyTorch di RAM CPU dan membiarkan
ONNX Runtime menjadi satu-satunya pemilik bobot di GPU. Pengaturan kedua mencegah
arena CUDA ONNX membulatkan setiap pertumbuhan buffer ke pangkat dua. Ini penting
untuk evaluasi paper dengan window 1.024 token, karena KV cache decoder bertambah
setiap token. Setelah mengubah pengaturan atau kode inference, recreate image:

```bash
docker compose build inference
docker compose up -d --force-recreate inference
```

Jangan mengubah `RETRO_BOOTSTRAP_DEVICE` kembali ke `cuda` pada GPU 16 GB kecuali
hasil `nvidia-smi` menunjukkan ruang cukup untuk model PyTorch, dua session ONNX,
dan KV cache sekaligus. Bootstrap CPU dapat menambah sedikit latency awal; nilainya
berlaku sama bagi RETRO dan PipeRAG sehingga pasangan evaluasinya tetap terkontrol.

Runtime vendor juga memakai jalur cached-attention bounded-memory. Jalur upstream
awalnya membangun positional bias dan causal mask penuh `L x L` pada setiap token,
meskipun cached decoder hanya memakai baris terakhir. Runtime lokal menghitung baris
`1 x L` tersebut secara langsung dengan nilai yang sama. Jika log lama menunjukkan
OOM pada node `positional_bias/Transpose`, rebuild image inference. Hash runtime akan
membuat artifact ONNX lama diekspor ulang otomatis; jangan menyalin kembali manifest
ONNX dari image/run sebelumnya.

Satu lock `artifacts/.benchmark-gpu.lock` mencegah dua benchmark dari direktori
output berbeda memakai GPU secara bersamaan. Lock dilepas otomatis ketika proses
selesai atau mati. Ini tidak dapat menghentikan proses lama yang dijalankan
sebelum mekanisme lock tersedia; identifikasi proses tersebut dengan
`nvidia-smi` dan `ps -fp <PID>` sebelum menghentikannya.

### Menjalankan benchmark secara detach

Load `.env` ke environment shell terlebih dahulu karena proses background mewarisi
environment saat command dijalankan. Setelah itu tambahkan `--detach` (atau `-d`):

```bash
set -a; source .env; set +a

uv run --no-sync python -m rag_benchmark benchmark \
  --config configs/experiments/its.yaml --stage all \
  --output artifacts/benchmarks/its --detach
```

Command langsung kembali dengan PID serta path log. Pantau proses dan output dengan:

```bash
cat artifacts/benchmarks/its/benchmark.pid
ps -fp "$(cat artifacts/benchmarks/its/benchmark.pid)"
tail -f artifacts/benchmarks/its/benchmark.log
```

Untuk menghentikan run secara normal, kirim `SIGTERM` ke PID yang tercatat:

```bash
kill -TERM "$(cat artifacts/benchmarks/its/benchmark.pid)"
```

Satu direktori output hanya boleh memiliki satu proses detach aktif. Gunakan direktori
output lain untuk run paralel; PID yang sudah tidak aktif akan digantikan pada run baru.

### Memulihkan permission index dari versi lama

Jika index pernah dibuat oleh container sebelum host menyiapkan direktori manifest,
direktori `artifacts/indexes/retro` dapat dimiliki `root` dan menghasilkan
`PermissionError` saat menulis `index-manifest.json`. Pulihkan ownership—jangan gunakan
`chmod 777`:

```bash
sudo chown -R "$(id -u):$(id -g)" artifacts/indexes/retro
chmod -R u+rwX artifacts/indexes/retro
```

Index FAISS yang sudah selesai tidak perlu dihapus. Jalankan kembali command benchmark;
service retrieval akan memakai cache index di disk lalu host menulis manifest yang tadi
gagal. Versi runner saat ini membuat direktori manifest sebagai user host sebelum build
index dimulai, sehingga ownership conflict ini tidak terulang pada direktori baru.

## 3. Acceptance full-run

- `docker compose ps` menunjukkan dua service sehat; hanya inference memakai GPU.
- Jumlah record prediksi sama dengan jumlah QA terjadwal dan error rate nol.
- Dua run dengan seed/config sama menghasilkan prediksi identik.
- RETRO fresh dan PipeRAG tanpa staleness cocok token/NLL dalam toleransi.
- `paper/pareto.{json,csv,png}` tersedia dan manifest mencatat seluruh hash.

Checkpoint upstream dominan Inggris dan tidak di-fine-tune pada ITS/UniQA;
kualitas Indonesia/Italia yang rendah harus dicatat sebagai keterbatasan model.
