#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import time
from collections import Counter
from collections import deque
from datetime import datetime
import socket
import threading
import os

import cv2 as cv
import numpy as np
import mediapipe as mp
import paho.mqtt.client as mqtt

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=480)
    parser.add_argument("--height", help='cap height', type=int, default=320)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.2)
    
    # MQTT arguments
    parser.add_argument("--mqtt_broker", help='MQTT broker address', type=str, default="broker.hivemq.com")
    parser.add_argument("--mqtt_port", help='MQTT port', type=int, default=1883)
    parser.add_argument("--mqtt_topic", help='MQTT topic', type=str, default="raspi/hcmus/car/gesture")
    parser.add_argument("--mqtt_log_topic", help='MQTT topic', type=str, default="raspi/hcmus/car/log")
    parser.add_argument("--enable_mqtt", help='Enable MQTT publishing', action='store_true')

    args = parser.parse_args()

    return args


# MQTT Setup Functions
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker successfully")
    else:
        print(f"Failed to connect to MQTT broker, return code {rc}")

def setup_mqtt(broker, port):
    try:
        # Fix for paho-mqtt version 2.0+ compatibility
        try:
            # Try new API first (paho-mqtt >= 2.0)
            client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, client_id="hand_gesture_controller")
        except:
            # Fallback to old API (paho-mqtt < 2.0)
            client = mqtt.Client("hand_gesture_controller")
        
        client.on_connect = on_connect
        client.connect(broker, port, 60)
        client.loop_start()
        return client
    except Exception as e:
        print(f"MQTT setup failed: {e}")
        return None

def check_gesture_stability(gesture_buffer, gesture_id, labels, threshold):
    """Check if a gesture is stable enough to publish"""
    if 0 <= gesture_id < len(labels):
        command = labels[gesture_id]
        gesture_buffer.append(command)
        
        # Count occurrences of each gesture in buffer
        gesture_counts = Counter(gesture_buffer)
        most_common = gesture_counts.most_common(1)
        
        if most_common and most_common[0][1] >= threshold:
            return most_common[0][0]  # Return the most stable gesture
    
    return None

def publish_gesture_command(mqtt_client, topic, gesture_id, labels, last_published, last_time, publish_interval):
    if mqtt_client is None:
        return last_published, last_time
    
    if 0 <= gesture_id < len(labels):
        command = labels[gesture_id]
        current_time = time.time()
        
        # Only publish if gesture actually changed
        if command != last_published:
            try:
                mqtt_client.publish(topic, command)
                print(f"Published: {command} to {topic}")
                return command, current_time
            except Exception as e:
                print(f"Failed to publish MQTT message: {e}")
    
    return last_published, last_time

def log_camera_issue(mqtt_client, topic, message):
    """Log camera issue and push to MQTT if enabled."""
    print(message)
    if mqtt_client is not None:
        try:
            mqtt_client.publish(topic, message)
            print(f"Published log: {message} to {topic}")
        except Exception as e:
            print(f"Failed to publish MQTT log: {e}")

wifi_disconnected = False

# Function to check internet connection
def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        return False


# Function to log WiFi disconnection offline
def log_wifi_disconnection_offline(message):
    log_dir = "logs"
    log_file = os.path.join(log_dir, "wifi_disconnection.log")

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Append the log message to the file
    with open(log_file, "a") as f:
        f.write(message + "\n")


# Function to check if MQTT client is connected
def is_mqtt_connected(mqtt_client):
    return mqtt_client.is_connected() if mqtt_client else False

# Function to push offline logs to MQTT
# Push logs stored in the offline log file to the MQTT broker
def push_offline_logs(mqtt_client, log_topic):
    log_dir = "logs"
    log_file = os.path.join(log_dir, "wifi_disconnection.log")

    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = f.readlines()

            remaining_logs = []
            for log in logs:
                if is_mqtt_connected(mqtt_client):
                    try:
                        mqtt_client.publish(log_topic + "/wifi", log.strip())
                        print(f"Pushed offline log: {log.strip()} to {log_topic}/wifi")
                    except Exception as e:
                        print(f"Failed to push log: {log.strip()} - {e}")
                        remaining_logs.append(log)  # Keep failed logs
                else:
                    print("MQTT client not connected. Retaining log.")
                    remaining_logs.append(log)

            # Write back remaining logs to the file
            with open(log_file, "w") as f:
                f.writelines(remaining_logs)
        except Exception as e:
            print(f"Failed to push offline logs: {e}")

# Update periodic_wifi_check to push offline logs upon reconnection
def periodic_wifi_check(interval, mqtt_client, log_topic):
    global wifi_disconnected
    while True:
        if not check_internet_connection():
            if not wifi_disconnected:
                wifi_disconnected = True
                final_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                final_message = f"[{final_time}] WiFi disconnected!"
                log_camera_issue(mqtt_client, log_topic + "/wifi", final_message)
                log_wifi_disconnection_offline(final_message)  # Log offline
        else:
            if wifi_disconnected:
                wifi_disconnected = False
                print("WiFi reconnected! Pushing offline logs...")
                push_offline_logs(mqtt_client, log_topic + "/wifi")  # Push offline logs
        time.sleep(interval)


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    mqtt_client = None
    if args.enable_mqtt:
        print(f"Setting up MQTT connection to {args.mqtt_broker}:{args.mqtt_port}")
        mqtt_client = setup_mqtt(args.mqtt_broker, args.mqtt_port)
        if mqtt_client:
            print(f"MQTT will publish to topic: {args.mqtt_topic}")
            print(f"Log will publish to topic: {args.mqtt_log_topic}")

            time.sleep(2)
            log_dir = "logs"
            log_file = os.path.join(log_dir, "wifi_disconnection.log")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    remaining_logs = f.readlines()
                if remaining_logs:
                    print("Offline logs detected at startup. Pushing...")
                    if is_mqtt_connected(mqtt_client):
                        push_offline_logs(mqtt_client, args.mqtt_log_topic)
                    else:
                        print("MQTT client not connected. Retaining log.")
        else:
            print("MQTT setup failed, continuing without MQTT")

    print("Camera opening")

    cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera with device index {cap_device}")
        return
    print("Camera opened successfully")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # MQTT throttling variables ############################################
    last_published_gesture = {"hand": None}
    last_publish_time = {"hand": 0}
    
    # Gesture confidence threshold
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for gesture recognition
    
    # Gesture stability tracking
    gesture_stability_buffer = {
        "hand": deque(maxlen=10)
    }  # Track last 10 gestures for hand only
    stable_gesture_threshold = 6  # Gesture must appear 6+ times out of 10 to be considered stable
    last_stable_gesture = {"hand": None}
    
    # Camera disconnection tracking
    camera_disconnected = False  # Flag to track if camera disconnection has been logged
    
    # Unknown gesture tracking
    unknown_gesture_start_time = None  # Track when unknown gesture started
    unknown_gesture_threshold = 2.0  # 2 seconds threshold for unknown gesture

    # WiFi disconnection tracking
    wifi_disconnected = False

    #  ########################################################################
    mode = 0

    # Start a thread for periodic WiFi check
    if args.enable_mqtt:
        wifi_check_thread = threading.Thread(target=periodic_wifi_check, args=(5, mqtt_client, args.mqtt_log_topic), daemon=True)
        wifi_check_thread.start()

    while True:
        if not cap.grab():
            print("Không thể grab frame!")
            break
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.retrieve()
        if not ret or not cap.isOpened():
            if not camera_disconnected:
                camera_disconnected = True
                print("Webcam disconnected! Waiting 10 seconds for webcam reconnection...")
                time.sleep(10)
                cap.release()
                cap = cv.VideoCapture(cap_device, cv.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
                    camera_disconnected = False
                    print('Webcam reconnected successfully!')
                    continue
                else:
                    final_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    final_message = f"[{final_time}] Webcam disconnected! Failed to reconnect webcam."
                    log_camera_issue(mqtt_client, args.mqtt_log_topic + "/webcam", final_message)
            break

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification with confidence threshold
                hand_sign_id, confidence = keypoint_classifier(pre_processed_landmark_list)
                
                # Check confidence threshold for unknown gesture detection
                if confidence >= CONFIDENCE_THRESHOLD:
                    current_hand_sign_id = hand_sign_id
                else:
                    current_hand_sign_id = 5  # Unknown gesture (index 5 in labels)
                
                # MQTT Publishing - Send gesture command only when stable
                if mqtt_client is not None:
                    # For unknown gestures, send "Stop" command only after 5 seconds
                    if current_hand_sign_id == 5:  # Unknown gesture
                        current_time = time.time()
                        
                        if unknown_gesture_start_time is None:
                            # Start tracking unknown gesture time
                            unknown_gesture_start_time = current_time
                        elif current_time - unknown_gesture_start_time >= unknown_gesture_threshold:
                            # Unknown gesture has lasted 5 seconds, send stop command
                            stop_command = "Stop"
                            if last_stable_gesture["hand"] != stop_command:
                                mqtt_client.publish(args.mqtt_topic, stop_command)
                                print(f"Unknown Gesture - Published: {stop_command} to {args.mqtt_topic}")
                                last_stable_gesture["hand"] = stop_command
                    else:
                        # Reset unknown gesture timer when we have a known gesture
                        unknown_gesture_start_time = None
                        
                        # Check hand gesture stability for known gestures
                        stable_hand_gesture = check_gesture_stability(
                            gesture_stability_buffer["hand"], current_hand_sign_id, 
                            keypoint_classifier_labels, stable_gesture_threshold
                        )
                        
                        # Publish only if stable gesture is different from last stable
                        if stable_hand_gesture and stable_hand_gesture != last_stable_gesture["hand"]:
                            mqtt_client.publish(args.mqtt_topic, stable_hand_gesture)
                            print(f"Stable Published: {stable_hand_gesture} to {args.mqtt_topic}")
                            last_stable_gesture["hand"] = stable_hand_gesture

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[current_hand_sign_id],
                    f"Conf: {confidence:.2f}",
                )
        else:
            # No hand detected - reset unknown gesture timer and send stop command
            unknown_gesture_start_time = None
            point_history.append([0, 0])
            
            # Send stop command when no hand is detected
            if mqtt_client is not None:
                stop_command = "Stop"
                # Only send stop if last published command was not already stop
                if last_stable_gesture["hand"] != stop_command:
                    mqtt_client.publish(args.mqtt_topic, stop_command)
                    print(f"No hand detected - Published: {stop_command} to {args.mqtt_topic}")
                    last_stable_gesture["hand"] = stop_command

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   confidence_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 44),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 26),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Display confidence score
    if confidence_text != "":
        cv.putText(image, confidence_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
