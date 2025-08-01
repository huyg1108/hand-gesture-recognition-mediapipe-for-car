#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test MQTT Receiver - Simulates ESP32 receiving commands
Chạy file này để test nhận tín hiệu từ gesture recognition
"""
import time
import paho.mqtt.client as mqtt

# MQTT Configuration
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "raspi/hcmus/car/gesture"  # Updated to match app.py
LOG_TOPIC = "raspi/hcmus/car/log"  # Added log topic from app.py
WIFI_LOG_TOPIC = "raspi/hcmus/car/log/wifi"  # Added WiFi log topic
ERROR_TOPIC = "car/error"  # Error topic for Node-RED
CLIENT_ID = "car_controller_test"

# Gesture to car command mapping
GESTURE_COMMANDS = {
    "Forward": {"description": "🚗 Di chuyển tiến", "action": "motor_forward()"},
    "Back": {"description": "🔄 Di chuyển lùi", "action": "motor_backward()"},
    "Left": {"description": "⬅️ Rẽ trái", "action": "turn_left()"},
    "Right": {"description": "➡️ Rẽ phải", "action": "turn_right()"},
    "Stop": {"description": "🛑 Dừng xe", "action": "motor_stop()"},
}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker successfully")
        print(f"📡 Subscribed to: {TOPIC}")
        print(f"📡 Subscribed to: {LOG_TOPIC}")
        print(f"📡 Subscribed to: {WIFI_LOG_TOPIC}")
        print(f"📡 Subscribed to: {ERROR_TOPIC}")
        client.subscribe(TOPIC)
        client.subscribe(LOG_TOPIC)
        client.subscribe(WIFI_LOG_TOPIC)
        client.subscribe(ERROR_TOPIC)
    else:
        print(f"❌ Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    try:
        command = msg.payload.decode('utf-8')
        topic = msg.topic
        
        print(f"\n📨 Received message:")
        print(f"   Topic: {topic}")
        print(f"   Command: {command}")
        print(f"   Time: {time.strftime('%H:%M:%S')}")
        
        if topic == TOPIC and command in GESTURE_COMMANDS:
            gesture_info = GESTURE_COMMANDS[command]
            print(f"   🎯 Action: {gesture_info['description']}")
            print(f"   🔧 Function: {gesture_info['action']}")
            
            # Simulate car control
            execute_car_command(command)
            
        elif topic == LOG_TOPIC:
            print(f"   📝 System log: {command}")
            
        elif topic == WIFI_LOG_TOPIC:
            print(f"   � WiFi log: {command}")
            
        elif topic == ERROR_TOPIC:
            print(f"   🚨 Error message: {command}")
            
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error processing message: {e}")

def execute_car_command(command):
    """Simulate ESP32 car control functions"""
    print(f"🤖 ESP32 executing: {command}")
    
    if command == "Forward":
        print("   🔧 digitalWrite(motor1_pin1, HIGH)")
        print("   🔧 digitalWrite(motor1_pin2, LOW)")
        print("   🔧 digitalWrite(motor2_pin1, HIGH)")
        print("   🔧 digitalWrite(motor2_pin2, LOW)")
    elif command == "Back":
        print("   🔧 digitalWrite(motor1_pin1, LOW)")
        print("   🔧 digitalWrite(motor1_pin2, HIGH)")
        print("   🔧 digitalWrite(motor2_pin1, LOW)")
        print("   🔧 digitalWrite(motor2_pin2, HIGH)")
    elif command == "Left":
        print("   🔧 digitalWrite(motor1_pin1, LOW)")
        print("   🔧 digitalWrite(motor1_pin2, HIGH)")
        print("   🔧 digitalWrite(motor2_pin1, HIGH)")
        print("   🔧 digitalWrite(motor2_pin2, LOW)")
    elif command == "Right":
        print("   🔧 digitalWrite(motor1_pin1, HIGH)")
        print("   🔧 digitalWrite(motor1_pin2, LOW)")
        print("   🔧 digitalWrite(motor2_pin1, LOW)")
        print("   🔧 digitalWrite(motor2_pin2, HIGH)")
    elif command == "Stop":
        print("   🔧 digitalWrite(motor1_pin1, LOW)")
        print("   🔧 digitalWrite(motor1_pin2, LOW)")
        print("   🔧 digitalWrite(motor2_pin1, LOW)")
        print("   🔧 digitalWrite(motor2_pin2, LOW)")

def main():
    print("🚗 Car Controller Test - MQTT Receiver")
    print("=" * 50)
    print(f"🔗 Connecting to MQTT broker: {BROKER}:{PORT}")
    
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, client_id=CLIENT_ID)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(BROKER, PORT, 60)
        client.loop_start()
        
        print("🎯 Waiting for gesture commands...")
        print("⌨️ Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping MQTT receiver...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        print("👋 Disconnected from MQTT broker")

if __name__ == "__main__":
    main()
