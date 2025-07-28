# 🤖 Hand Gesture Car Control - Test Guide

## 📋 Tổng quan hệ thống

Hệ thống nhận diện cử chỉ tay để điều khiển xe gồm:
- **Camera laptop** → **Gesture Recognition** → **MQTT** → **ESP32** → **Car Control**

## 🎯 Các cử chỉ được nhận diện:
1. **Forward** - Cử chỉ tiến (✋ mở bàn tay)
2. **Back** - Cử chỉ lùi 
3. **Left** - Cử chỉ rẽ trái
4. **Right** - Cử chỉ rẽ phải  
5. **Stop** - Cử chỉ dừng (✊ nắm tay)

## 🔧 Cài đặt môi trường

### 1. Cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### 2. Kiểm tra camera:
```bash
# Test camera hoạt động
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

## 🧪 Cách test hệ thống

### Step 1: Test chỉ nhận diện (không MQTT)
```bash
# Chạy gesture recognition cơ bản
python app.py
```
- Quan sát cử chỉ được nhận diện trên màn hình
- Nhấn ESC để thoát

### Step 2: Test với MQTT receiver
```bash
# Terminal 1: Chạy MQTT receiver (giả lập ESP32)
python test_mqtt_receiver.py

# Terminal 2: Chạy gesture recognition với MQTT
python app.py --enable_mqtt
```

### Step 3: Test với custom MQTT settings
```bash
python app.py --enable_mqtt --mqtt_broker "your_broker" --mqtt_topic "custom/topic"
```

## 📡 MQTT Configuration

### Default settings:
- **Broker**: broker.hivemq.com
- **Port**: 1883  
- **Topic**: raspi/test
- **Finger Topic**: car/control/finger

### Custom MQTT parameters:
```bash
python app.py --enable_mqtt \
  --mqtt_broker "broker.hivemq.com" \
  --mqtt_port 1883 \
  --mqtt_topic "raspi/test"
```

## 🔍 Debugging và kiểm tra

### 1. Kiểm tra MQTT connection:
Khi chạy `python app.py --enable_mqtt`, bạn sẽ thấy:
```
Setting up MQTT connection to broker.hivemq.com:1883
Connected to MQTT broker successfully
MQTT will publish to topic: car/control
```

### 2. Kiểm tra gesture detection:
- Quan sát terminal sẽ hiển thị: `📡 Published: Forward to car/control`
- MQTT receiver sẽ nhận và hiển thị commands

### 3. Troubleshooting:
- **Camera không mở được**: Thay đổi `--device 1` hoặc `--device 2`
- **MQTT lỗi**: Kiểm tra internet connection
- **Gesture không chính xác**: Điều chỉnh lighting và góc camera

## 🚀 Chuyển đổi lên Raspberry Pi

### Các thay đổi cần thiết:

1. **Camera setup cho RPi**:
```python
# Thay đổi trong app.py
cap = cv.VideoCapture(0)  # USB camera trên RPi
# hoặc
cap = cv.VideoCapture('/dev/video0')  # Specific device
```

2. **MQTT broker local** (tùy chọn):
```bash
# Cài đặt Mosquitto trên RPi
sudo apt install mosquitto mosquitto-clients
sudo systemctl start mosquitto

# Sử dụng local broker
python app.py --enable_mqtt --mqtt_broker "localhost"
```

3. **Optimize cho RPi**:
```python
# Thêm vào app.py để giảm CPU usage
parser.add_argument("--width", type=int, default=640)  # Giảm resolution
parser.add_argument("--height", type=int, default=480)
```

## 📊 ESP32 Code Structure

```cpp
// ESP32 sẽ nhận MQTT messages như sau:
void callback(char* topic, byte* payload, unsigned int length) {
  String command = String((char*)payload).substring(0, length);
  
  if (command == "Forward") {
    moveForward();
  } else if (command == "Back") {
    moveBackward();
  } else if (command == "Left") {
    turnLeft();
  } else if (command == "Right") {
    turnRight();
  } else if (command == "Stop") {
    stopMotors();
  }
}
```

## ⚠️ Lưu ý quan trọng

1. **Performance**: Gesture recognition có thể chậm trên laptop yếu
2. **Lighting**: Cần ánh sáng tốt để nhận diện chính xác
3. **Distance**: Giữ tay cách camera 50-100cm
4. **Stability**: Giữ cử chỉ ổn định 1-2 giây để tránh nhiễu

## 🎛️ Tùy chỉnh thêm

### Thêm gesture mới:
1. Edit `model/keypoint_classifier/keypoint_classifier_label.csv`
2. Train lại model nếu cần
3. Update ESP32 code để handle command mới

### Thay đổi MQTT topic structure:
```python
# Có thể custom theo format:
# car/control/direction  (Forward, Back, Left, Right, Stop)
# car/control/speed      (Fast, Slow)
# car/control/mode       (Manual, Auto)
```
