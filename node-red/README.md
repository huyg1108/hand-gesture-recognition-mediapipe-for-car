# 🚗 Hand Gesture Car Control System
## *Complete IoT Solution với Node-RED Dashboard + ThingSpeak Analytics*

## 📋 Tổng quan hệ thống:

Hệ thống điều khiển xe bằng cử chỉ tay hoàn chỉnh với real-time monitoring, cloud analytics và comprehensive logging:

```
[Hand Gesture App] → [MQTT] → [Node-RED Dashboard] → [ThingSpeak Cloud]
       ↓                ↓            ↓                    ↓
   MediaPipe       broker.hivemq    Web Interface      Analytics
   Recognition     .com:1883        Dashboard          Storage
```

### **🎯 Core Technologies:**
- **🖐️ MediaPipe** - Hand gesture recognition với AI
- **📡 MQTT** - Real-time communication protocol  
- **🎛️ Node-RED** - Visual flow programming + dashboard
- **☁️ ThingSpeak** - Cloud analytics & visualization
- **📊 Real-time Monitoring** - System health tracking

## ✨ Tính năng nổi bật:

### **🎮 Gesture Control:**
- **5 Gestures**: ⬆️Forward, ⬇️Backward, ⬅️Left, ➡️Right, 🛑Stop
- **AI Recognition**: MediaPipe với confidence threshold
- **Stability Check**: Anti-jitter với gesture buffering
- **Unknown Gesture**: Auto-stop sau 2 giây

### **📊 Dashboard Components:**
- **Car Controls**: 5 gesture buttons + manual override
- **Command Logs**: Real-time sent/received commands
- **System Status**: Raspberry Pi health monitoring
- **Error Monitoring**: Comprehensive error tracking
- **Log Management**: Clear logs + auto-cleanup

### **📡 MQTT Topics Structure:**
```
raspi/hcmus/car/gesture        # Gesture commands
raspi/hcmus/car/log/system     # System lifecycle 
raspi/hcmus/car/log/webcam     # Camera status
raspi/hcmus/car/log/wifi       # Network connectivity
```

### **☁️ ThingSpeak Integration (3-Field):**
- **Field 1**: Severity Level (1=Low, 2=Medium, 3=High)
- **Field 2**: Combined Error Message
- **Field 3**: ISO Timestamp
- **Rate Limiting**: 15-second intervals
- **Auto-Upload**: Critical errors only

### **🛡️ System Monitoring:**
- **Raspberry Pi Status**: Camera + WiFi + System health
- **Error Detection**: Keyword-based error categorization
- **Offline Resilience**: Local backup + recovery
- **Graceful Shutdown**: Comprehensive cleanup notifications

## 🚀 Quick Start Guide:

### **Step 1: Setup Node-RED**
```bash
# Install Node-RED globally
npm install -g node-red

# Install required nodes
npm install -g node-red-dashboard

# Start Node-RED server
node-red
```

### **Step 2: Import Dashboard Flow**
```bash
# 1. Open Node-RED: http://localhost:1880
# 2. Menu (☰) → Import → Upload file
# 3. Select: final-complete-flow.json
# 4. Click "Deploy" button
# 5. Success! ✅
```

### **Step 3: Access Dashboard**
```bash
# Dashboard URL:
http://localhost:1880/ui

# Expected: Complete car control interface
```

### **Step 4: Run Hand Gesture App**
```bash
cd hand-gesture-recognition-mediapipe-for-car

# Install dependencies
pip install -r requirements.txt

# Run with MQTT enabled
python app.py --enable_mqtt

# Expected: Camera opens + MQTT connection
```

## 📊 Dashboard Layout:

```
🚗 Car Control Dashboard
│
├── 🎮 Car Controls
│   ├── ⬆️ Forward  ⬇️ Backward  ⬅️ Left  ➡️ Right  🛑 Stop
│   └── 🗑️ Clear All Logs
│
├── 📋 Command Logs  
│   ├── 📤 Sent Commands     │ 📥 Received Commands
│   └── [Real-time display]  │ [MQTT subscriber logs]
│
├── 🍓 System Status (Raspberry Pi Monitoring)
│   ├── 📹 Camera: ✅ Active / ❌ Error
│   ├── 📶 WiFi: ✅ Connected / ❌ Disconnected  
│   ├── 🕒 Last Update: [Timestamp]
│   └── 📋 Recent System Logs (6 latest entries)
│
└── ⚠️ Error Monitoring
    ├── � High Priority Errors
    ├── 🟡 Medium Priority Warnings
    ├── 🟢 Low Priority Info
    └── ☁️ Auto-sync to ThingSpeak
```

## 🔧 Project Structure:

```
AIOT/
├── 📄 final-complete-flow.json          # ⭐ Main Node-RED flow (IMPORT THIS)
├── 📚 COMPLETE_SYSTEM_GUIDE.md          # Comprehensive documentation
├── 🧪 test_error_logging.py             # Error scenario testing
├── 📊 slides_app_logging.md             # Presentation slides
├── 🛠️ create_presentation_graphs.py     # Analytics graphs
├── hand-gesture-recognition-mediapipe-for-car/
│   ├── 🎯 app.py                        # Main gesture recognition app
│   ├── 📋 requirements.txt              # Python dependencies
│   ├── 🤖 model/                        # AI models directory
│   │   └── keypoint_classifier/         # Gesture classification model
│   ├── 🛠️ utils/                        # Utility functions
│   └── 📝 test_mqtt_receiver.py         # MQTT testing utility
└── 📖 README.md                         # This comprehensive guide
```

## 🧪 Testing & Validation:

### **🎯 Test Complete System:**
```bash
# 1. Test Node-RED Dashboard
# Import final-complete-flow.json → Deploy → Open http://localhost:1880/ui
# Expected: All 4 dashboard sections visible and functional

# 2. Test Hand Gesture App
cd hand-gesture-recognition-mediapipe-for-car
python app.py --enable_mqtt
# Expected: Camera opens + MQTT logs in dashboard

# 3. Test Error Scenarios
python test_error_logging.py
# Expected: Error monitoring updates + ThingSpeak data
```

### **🔍 Verification Checklist:**
- ✅ **Dashboard responsive**: All buttons work instantly
- ✅ **Real-time logs**: Commands appear immediately  
- ✅ **System monitoring**: Pi status updates correctly
- ✅ **Error detection**: Test errors trigger alerts
- ✅ **ThingSpeak sync**: Critical errors upload to cloud
- ✅ **Gesture recognition**: Hand movements control car
- ✅ **Offline resilience**: Logs saved during disconnection

### **📊 Expected Behaviors:**

#### **Normal Operation:**
```
Dashboard: All ✅ green status indicators
Logs: Startup messages → gesture commands → system heartbeat
ThingSpeak: No data (only errors uploaded)
Console: Clean gesture recognition output
```

#### **Error Scenarios:**
```
Camera disconnect → ❌ Camera status → ThingSpeak severity=3
WiFi loss → ❌ WiFi status → Offline log backup
App shutdown → All ❌ status → Graceful cleanup notifications
```

## � ThingSpeak Cloud Analytics:

### **📈 3-Field Data Structure:**
```javascript
// Optimized for analytics
Field 1: Severity Level    // 1=Low, 2=Medium, 3=High/Critical
Field 2: Error Message     // "[CATEGORY] Description"  
Field 3: ISO Timestamp     // "2025-08-02T10:30:45.123Z"
```

### **🔍 Sample Data:**
```json
{
  "field1": 3,                                    // High severity
  "field2": "[CAMERA_ISSUE] Camera disconnected - System shutdown",
  "field3": "2025-08-02T15:30:45.123Z"
}
```

### **📊 Analytics Capabilities:**
- **📈 Severity Trends**: Track error frequency over time
- **🔍 Error Categories**: Camera vs WiFi vs System issues  
- **⏰ Timeline Analysis**: Peak error times identification
- **📋 MATLAB Integration**: Advanced data processing

### **⚙️ Configuration:**
```javascript
// ThingSpeak settings trong Node-RED
API_ENDPOINT: "https://api.thingspeak.com/update"
RATE_LIMIT: 15000ms  // 15-second intervals
AUTO_UPLOAD: Critical errors only (severity >= 2)
```

## 🤖 AI Gesture Recognition:

### **🖐️ Supported Gestures:**
```python
Gesture Classes:
0: Forward    # ✊ Closed fist
1: Backward   # 🖐️ Open palm  
2: Left       # 👈 Point left
3: Right      # 👉 Point right
4: Stop       # ✋ Stop gesture
5: Unknown    # Unrecognized → Auto-stop after 2s
```

### **🎯 AI Configuration:**
```python
# Performance optimized settings
CONFIDENCE_THRESHOLD = 0.5      # Balance accuracy vs speed
GESTURE_STABILITY = 3           # Require 3 consistent readings
FRAME_SKIP = 1                  # Process every 2nd frame
UNKNOWN_TIMEOUT = 2.0           # Auto-stop after 2 seconds
```

### **📡 MQTT Publishing Logic:**
```python
# Intelligent gesture filtering
1. MediaPipe detection → confidence check
2. Stability buffer → prevent jitter  
3. Change detection → only publish new gestures
4. Unknown handling → timeout safety mechanism
```

## 🛡️ System Health Monitoring:

### **📊 Raspberry Pi Status Tracking:**
- **📹 Camera**: Initialization, disconnection, recovery attempts
- **📶 WiFi**: Connection status, offline backup, reconnection
- **🖥️ System**: Startup, heartbeat, graceful shutdown
- **⚠️ Errors**: Real-time detection với keyword analysis

### **🔄 Recovery Mechanisms:**
```python
Camera Disconnect:
├── 10-second wait period
├── Automatic reconnection attempt  
├── Settings restoration
└── Failure → Error logging + status update

WiFi Disconnect:
├── Offline log backup to local file
├── Periodic connection checks (5s)
├── Auto-recovery on reconnection
└── Batch upload of offline logs
```

### **🚨 Error Categories:**
- **🔴 High**: System shutdown, camera failure, critical errors
- **🟡 Medium**: Network timeouts, video stream interrupts  
- **🟢 Low**: Info messages, successful operations

## 🎯 Success Indicators:

### **✅ Dashboard Working:**
```
✓ All 4 sections visible and responsive
✓ Buttons trigger immediate command logs  
✓ System status shows real-time Pi health
✓ Error monitoring displays categorized alerts
✓ Clear logs function resets all displays
✓ No page refresh required for updates
```

### **✅ MQTT Integration:**
```
✓ Connection to broker.hivemq.com:1883
✓ Publishing to raspi/hcmus/car/* topics
✓ Real-time message flow dashboard ↔ app
✓ Gesture commands reach car controller
✓ System logs update Node-RED displays
```

### **✅ ThingSpeak Analytics:**
```
✓ Critical errors auto-upload (severity ≥ 2)
✓ 3-field structure: [Severity, Message, Timestamp]
✓ Rate limiting respected (15-second intervals)
✓ Data visible in ThingSpeak channel
✓ Analytics charts available for trends
```

### **✅ AI Gesture Recognition:**
```
✓ MediaPipe hand detection active
✓ 5 gesture classes recognized accurately
✓ Stability filtering prevents false triggers
✓ Unknown gestures safely timeout to Stop
✓ MQTT commands reflect hand movements
```

## 🚨 Troubleshooting Guide:

### **🔧 Dashboard Issues:**

#### **Problem**: Dashboard shows blank page
```bash
Solution:
1. Check Node-RED dashboard installed: npm list -g node-red-dashboard
2. Verify import: Menu → Flows → Check "main_tab" exists
3. Deploy: Red "Deploy" button → Wait for success
4. Refresh browser: Ctrl+F5 or clear cache
```

#### **Problem**: Buttons don't work
```bash
Solution:
1. Check all nodes in same flow tab
2. Verify MQTT node connections (no broken links)
3. Debug: Enable debug nodes → Check console output
4. Re-import final-complete-flow.json if needed
```

### **🔧 MQTT Connection Issues:**

#### **Problem**: MQTT not connecting
```bash
Solution:
1. Check internet: ping broker.hivemq.com
2. Firewall: Allow port 1883 outbound
3. Test: python test_error_logging.py
4. Verify credentials: No auth required for public broker
```

#### **Problem**: Messages not received
```bash
Solution:
1. Check topic spelling: raspi/hcmus/car/gesture
2. Verify QoS settings: Default 0 in Node-RED
3. Debug MQTT in: Enable output → Check message flow
4. Test with external MQTT client (MQTT Explorer)
```

### **🔧 Camera/AI Issues:**

#### **Problem**: Camera won't open
```bash
Solution:
1. Check camera permissions: Device Manager → Camera
2. Close other camera apps: Skype, Zoom, etc.
3. Try different device index: --device 1
4. Install OpenCV: pip install opencv-python
```

#### **Problem**: Gestures not recognized
```bash
Solution:
1. Check lighting: Ensure good hand visibility
2. Adjust confidence: Lower --min_detection_confidence
3. Hand positioning: Keep hand in center frame
4. Check model files: model/keypoint_classifier/*.tflite
```

### **🔧 ThingSpeak Issues:**

#### **Problem**: No data uploading
```bash
Solution:
1. Verify API key in Node-RED ThingSpeak node
2. Check rate limiting: 15s minimum between uploads
3. Test connection: curl ThingSpeak API endpoint
4. Trigger real error: python test_error_logging.py
```

#### **Problem**: Wrong data format
```bash
Solution:
1. Check 3-field structure: [severity, message, timestamp]
2. Verify JSON formatting in Node-RED function
3. Monitor debug console for format errors
4. Update ThingSpeak channel field names
```

## 👨‍💻 Development & Customization:

### **🎨 Dashboard Customization:**
```javascript
// Modify button appearance in ui_button nodes
Button Colors: {
  Forward:  "#4CAF50",   // Green
  Backward: "#FF9800",   // Orange  
  Left:     "#2196F3",   // Blue
  Right:    "#9C27B0",   // Purple
  Stop:     "#F44336"    // Red
}

// Update dashboard layout
Group Order: Car Controls(1) → Command Logs(2) → System Status(3) → Error Monitoring(4)
```

### **🔧 MQTT Customization:**
```javascript
// Change broker/topics trong MQTT nodes
MQTT_BROKER: "your-mqtt-broker.com"
GESTURE_TOPIC: "your/car/gesture"  
LOG_TOPICS: "your/car/log/{system,webcam,wifi}"

// Modify command mappings
Gesture Commands: ["Forward", "Backward", "Left", "Right", "Stop"]
```

### **☁️ ThingSpeak Customization:**
```javascript
// Update API configuration in ThingSpeak Formatter node
API_KEY: "YOUR_THINGSPEAK_API_KEY"
CHANNEL_ID: your_channel_id
FIELD_MAPPING: {
  field1: severity_level,
  field2: combined_message,
  field3: iso_timestamp
}
```

### **🤖 AI Model Customization:**
```python
# Update gesture recognition model
Model Path: model/keypoint_classifier/keypoint_classifier.tflite
Label File: model/keypoint_classifier/keypoint_classifier_label.csv

# Retrain với custom gestures:
1. Collect new gesture data: Modify logging_csv() mode
2. Train model: Use keypoint_classification.ipynb
3. Replace model files: .tflite và .csv
4. Update gesture count in app.py
```

## 🔗 Integration Examples:

### **🚗 Physical Car Integration:**
```python
# Add car control hardware interface
import RPi.GPIO as GPIO

def control_car_motors(command):
    if command == "Forward":
        GPIO.output(MOTOR_PIN_1, GPIO.HIGH)
    elif command == "Stop":
        GPIO.output(MOTOR_PIN_1, GPIO.LOW)
    # Add more motor controls...
```

### **📱 Mobile App Integration:**
```javascript
// Add mobile notifications
const notification_node = {
    "type": "function",
    "func": `
        if (msg.payload.severity >= 2) {
            // Send push notification
            msg.notification = {
                title: "Car Error Alert",
                body: msg.payload.message
            };
        }
        return msg;
    `
}
```

### **🏠 Home Automation:**
```javascript
// Integrate với Home Assistant
const homeassistant_node = {
    "type": "ha-api", 
    "service": "automation.trigger",
    "data": {
        "entity_id": "automation.car_emergency_stop",
        "variables": {
            "error_type": "{{payload.type}}",
            "severity": "{{payload.severity}}"
        }
    }
}
```

## 📚 Additional Resources:

### **📖 Documentation:**
- **[COMPLETE_SYSTEM_GUIDE.md](COMPLETE_SYSTEM_GUIDE.md)** - Detailed technical documentation
- **[slides_app_logging.md](slides_app_logging.md)** - Presentation slides về logging system
- **[Node-RED Documentation](https://nodered.org/docs/)** - Official Node-RED docs
- **[ThingSpeak API Guide](https://thingspeak.com/docs)** - ThingSpeak integration guide

### **🧪 Testing Tools:**
- **`test_error_logging.py`** - Comprehensive error scenario testing
- **`create_presentation_graphs.py`** - Analytics visualization tools
- **Node-RED Debug Panel** - Real-time flow debugging
- **MQTT Explorer** - External MQTT testing tool

### **🎓 Learning Path:**
1. **Beginner**: Import flow → Test dashboard → Basic customization
2. **Intermediate**: Modify MQTT topics → Custom error handling
3. **Advanced**: AI model training → Hardware integration → Production deployment

## 🏆 Project Achievements:

### **✨ Technical Highlights:**
- ✅ **Complete IoT stack**: Edge AI → MQTT → Dashboard → Cloud
- ✅ **Production-ready logging**: 4-layer monitoring với offline backup
- ✅ **Optimized performance**: Frame skipping, gesture stability, rate limiting
- ✅ **Graceful error handling**: Auto-recovery mechanisms
- ✅ **Scalable architecture**: Modular Node-RED flows
- ✅ **Real-time analytics**: ThingSpeak cloud integration

### **🎯 Perfect for:**
- **🎓 IoT Education**: Complete learning project
- **🚗 Autonomous Vehicles**: Gesture control interface
- **🏠 Smart Home**: Hand gesture device control  
- **🏭 Industrial Applications**: Touchless machine control
- **♿ Accessibility**: Alternative input methods

---

## 🚀 **Get Started Now!**

### **⚡ Quick Deploy (5 minutes):**
```bash
# 1. Clone & setup
git clone [repository]
cd AIOT/node-red

# 2. Start Node-RED  
node-red

# 3. Import flow
# Open: http://localhost:1880
# Import: final-complete-flow.json
# Deploy!

# 4. Open dashboard
# URL: http://localhost:1880/ui
# ✅ Success!
```

### **📞 Support:**
- **🐛 Issues**: Create GitHub issue với logs
- **💡 Features**: Submit pull request
- **❓ Questions**: Check troubleshooting guide first

**Happy building! 🎉**

---

**Main File: `final-complete-flow.json` - Contains everything you need! 🎯**
