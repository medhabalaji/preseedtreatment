# Chickpea Germination Intelligence

A Python Flask dashboard for chickpea germination monitoring. The main dashboard focuses on germination-relevant environmental conditions, while seed image monitoring remains available as a feature.

- Temperature/humidity simulator
- Chickpea germination suitability score
- Moisture stress and vapor pressure deficit
- Treated vs control chickpea image monitoring
- ESP32-CAM or webcam upload path

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Environment simulator

The app simulates temperature and humidity until real sensors are connected. It calculates:

- Germination suitability
- Temperature score
- Humidity score
- Moisture stress
- Vapor pressure deficit
- Risk label for cold delay, heat stress, drying stress, or excess moisture

## Seed image monitoring feature

Place both chickpea batches in one stable camera view. The treated batch must stay on the left and the control batch must stay on the right. Open the dashboard, click `Start Camera`, set the chickpeas per batch, then capture a single frame or start the auto monitor.

For a 2-3 day comparison, keep lighting, distance, tray position, and capture interval consistent. The dashboard compares germination speed in percentage points per day and growth speed in white radical pixels per day.

## OpenCV webcam monitoring

```powershell
python webcam_monitor.py --seed-count 30 --interval 20
```

## Upload an ESP32-CAM frame

Send raw JPEG bytes to the upload endpoint:

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:5000/upload_image?seed_count=30" -ContentType image/jpeg -InFile sample.jpg
```

For ESP32-CAM, mount the camera above the two chickpea trays so both batches are visible in the same frame. Use the computer's LAN IP instead of `localhost`, for example:

```text
http://10.27.223.43:5000/upload_image?seed_count=30
```

The app saves original captures in `static/captures/`, analyzed overlays in `static/overlays/`, and growth logs in `growth_logs.db`.

## Timelapse

After auto monitoring has collected at least two frames, click `Create Timelapse` on the dashboard. The app builds an MP4 from overlay frames and saves it in `static/timelapses/`.
