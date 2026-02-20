# Road-Anomaly-Detection
Real-time road anomaly detection system using Raspberry Pi

RoadGuard: Real-Time Road Anomaly Detection System
A real-time road anomaly detection system built using YOLOv8 and deployed on a Raspberry Pi. The system detects road anomalies such as potholes, cracks, open manholes, and rutting instantly using a camera module. The detected frames along with the location of the anomaly are automatically sent to a Telegram bot for instant monitoring and reporting — enabling authorities to act quickly.
The model was trained on Google Colab using an NVIDIA A100 GPU for 80 epochs and exported to ONNX format for lightweight edge deployment on the Raspberry Pi. The system is also capable of detecting anomalies in low-light and night conditions, making it reliable for 24/7 road monitoring.
