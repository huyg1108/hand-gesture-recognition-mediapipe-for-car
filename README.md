# CSC16106 - Project 2: Hand Gesture Car Control

## System Overview

The hand gesture recognition system for car control consists of:
- **Webcam** $\rightarrow$ **Gesture Recognition** $\rightarrow$ **MQTT** $\rightarrow$ **ESP32** $\rightarrow$ **Car Control**

## Recognized Gestures:
1. **Forward** - Car moves forward (open hand)
2. **Back** - Car moves backward (fist)
3. **Left** - Car turns left (thumb held horizontally, pointing to the side)
4. **Right** - Car turns right (thumb and index finger)
5. **Stop** - Car stops (index and pinky finger)

**Note**: These gestures are configured and tested for the right hand. The system has not yet been tested with the left hand.

## Environment Setup

* Python 3.7.3
* Install necessary libraries:
```bash
pip install -r requirements.txt
```

## How to Run

### Recognition Only (without MQTT)
```bash
python app.py
```
- Observe the recognized gesture on the screen.
- Press ESC to exit.

### Recognition with MQTT Receiver
```bash
python app.py --enable_mqtt
```

### Custom MQTT settings
```bash
python app.py --enable_mqtt --mqtt_broker "your_broker" --mqtt_topic "custom/topic"
```

## MQTT Configuration

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

## Optimize for RPi:

* **Reduce camera resolution**: Lower the processing resolution (e.g., 320x240) in the app.py script to decrease CPU load.
* **Display in grayscale**: Convert the preview window to grayscale to save rendering resources. The actual image processing can still use the RGB data for accuracy.
* **Lower system display resolution**: Reduce the Raspberry Pi's global screen resolution in the OS settings to free up overall system resources.

## ESP32 Code Structure

The ESP32 will receive MQTT messages as follows:
```cpp
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

## Important Notes

1. **Performance**: Gesture recognition can be slow on low-end hardware.
2. **Lighting**: Good lighting is required for accurate gesture recognition.
3. **Distance**: Keep your hand 50-100cm away from the camera.
4. **Stability**: Hold the gesture steady for 1-2 seconds to avoid noise.

## Further Customization

### Adding a New Gesture:

1.  **Edit the Label File:** First, add the name for your new gesture to the `model/keypoint_classifier/keypoint_classifier_label.csv` file. Take note of the number assigned to this new gesture label.

2.  **Collect Landmark Data for the New Gesture:**
    *   Run the application in default mode: `python app.py`.
    *   Press the **'K'** key on your keyboard to enter keypoint logging mode.
    *   Position your hand in front of the camera to form the new gesture.
    *   Press the number key that corresponds to the label you just created (e.g., if your new gesture is number 6 in the CSV, press '6'). This will save the current hand landmarks for that gesture. Repeat this process multiple times to build a good dataset.

3.  **Retrain the Model:** Run the training script to retrain the model with the newly added dataset.

4.  **Update ESP32 Code:** Modify the ESP32's `callback` function to handle the new command string that will be sent for your new gesture.

# Contributors:

* Pham Quang Duy
* Ngo Hoang Nam Hung
* Trieu Gia Huy
* Nguyen Gia Nguyen
