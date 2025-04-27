# 🐔 Chicken House Monitoring System

![System Diagram](https://example.com/your-diagram.png)

A local-network-based detection and monitoring system for chicken houses.  
It combines computer vision, sensor data, and a digital dashboard for real-time insights — all running securely offline.

## 🛠️ How It Works

- **Camera** streams a live video feed over HTTP (Wi-Fi).
- **Synology NAS** hosts a **Docker container** running a **Python script** that:
  - Processes the live feed with **Computer Vision** to count chickens and humans.
  - Exposes a local **HTTP endpoint** serving the detection results as **JSON**.
- **ESP32 Sensor Node** collects:
  - **Temperature** readings.
  - **Door status** (open/closed).
  - It also serves its data over a **local HTTP endpoint**.
- **ESP32 Dashboard Controller**:
  - Fetches detection data and sensor data.
  - Displays everything on a connected **digital screen**.

All components communicate exclusively over the local Wi-Fi network — no internet/cloud involved.

## 📈 System Overview Diagram

(Click the image above or view the live diagram [here](sensors/diagram.png))
