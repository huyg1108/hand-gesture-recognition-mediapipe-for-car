#include <WiFiManager.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <Arduino.h>
#include <U8g2lib.h>
#include <Wire.h>

// Chân motor
const int motor1Pin1 = 16; 
const int motor1Pin2 = 17; 
const int enable1Pin = 4; 

const int motor2Pin1 = 18; 
const int motor2Pin2 = 19; 
const int enable2Pin = 5; 

const int freq = 30000;
const int pwmChannel = 0;
const int resolution = 8;
int dutyCycleLeft = 250;   
int dutyCycleRight = 255; 

const char* mqtt_server = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_topic = "raspi/hcmus/car/gesture"; 
const char* mqtt_topic_log = "raspi/hcmus/car/log";

// Các tham số chung
const int centerX = 64;
const int centerY = 32;
const int size = 27;  // chiều dài thân
const int head = 20;  // kích thước đầu mũi tên
const int thick = 10;  // độ dày thân
String currentAction = "STOP";

WiFiClient espClient;
PubSubClient client(espClient);

U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

void setup_wifi() {
  WiFiManager wm;
  // wm.resetSettings();
  bool res = wm.autoConnect("Controlling car");
  
  if (!res) {
    Serial.println("WiFi Failed to connect. Restarting...");
    ESP.restart();
  }
  Serial.println("WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setupMotors() {
  pinMode(motor1Pin1, OUTPUT);
  pinMode(motor1Pin2, OUTPUT);
  pinMode(enable1Pin, OUTPUT);
  pinMode(motor2Pin1, OUTPUT);
  pinMode(motor2Pin2, OUTPUT);
  pinMode(enable2Pin, OUTPUT);

  // ledcAttachChannel(enable1Pin, freq, resolution, pwmChannel);
  // ledcAttachChannel(enable2Pin, freq, resolution, pwmChannel);
  // ledcWrite(enable1Pin, dutyCycleLeft);   
  // ledcWrite(enable2Pin, dutyCycleRight); 

  //Test
  Serial.print("Finished set up Motor");
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";

  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.println("📥 MQTT Message Received:");
  Serial.print("Topic: ");
  Serial.println(topic);
  Serial.print("Payload: ");
  Serial.println(message);
  if (message == "Forward") goForward();
  else if (message == "Back") goBackward();
  else if (message == "Left") turnLeft();
  else if (message == "Right") turnRight();
  else if (message == "Stop") stop();
  else stop();
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("ESP32CheckClient")) {
      Serial.println("Connected!");
      client.subscribe(mqtt_topic);
      Serial.print("Subscribed to: ");
      Serial.println(mqtt_topic);
    } else {
      Serial.print("Failed (code ");
      Serial.print(client.state());
      Serial.println("), retrying in 5s...");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(9600);
  u8g2.begin();
  setup_wifi();
  setupMotors();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  u8g2.clearBuffer();
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
  drawAction(currentAction);
  u8g2.sendBuffer();
}

void drawAction(String action) {
  u8g2.clearBuffer();
  if (action == "FORWARD") drawArrowForward();
  else if (action == "BACK") drawArrowBack();
  else if (action == "LEFT") drawArrowLeft();
  else if (action == "RIGHT") drawArrowRight();
  else drawStopSign(64, 32, 20);
  u8g2.sendBuffer();
}

// Hàm vẽ line dày
void drawThickLine(int x1, int y1, int x2, int y2, int thickness) {
  if (x1 == x2) {
    u8g2.drawBox(x1 - thickness/2, min(y1,y2), thickness, abs(y2-y1));
  } else if (y1 == y2) {
    u8g2.drawBox(min(x1,x2), y1 - thickness/2, abs(x2-x1), thickness);
  }
}

// Mũi tên lùi
void drawArrowBack() {
  drawThickLine(centerX, centerY+size/2, centerX, centerY-size/2, thick);
  u8g2.drawTriangle(centerX-head/2, centerY-size/2, centerX+head/2, centerY-size/2, centerX, centerY-size/2-head);
}

// Mũi tên tới
void drawArrowForward() {
  drawThickLine(centerX, centerY-size/2, centerX, centerY+size/2, thick);
  u8g2.drawTriangle(centerX-head/2, centerY+size/2, centerX+head/2, centerY+size/2, centerX, centerY+size/2+head);
}

// Mũi tên phải
void drawArrowRight() {
  drawThickLine(centerX+size/2, centerY, centerX-size/2, centerY, thick);
  u8g2.drawTriangle(centerX-size/2, centerY-head/2, centerX-size/2, centerY+head/2, centerX-size/2-head, centerY);
}

// Mũi tên trái
void drawArrowLeft() {
  drawThickLine(centerX-size/2, centerY, centerX+size/2, centerY, thick);
  u8g2.drawTriangle(centerX+size/2, centerY-head/2, centerX+size/2, centerY+head/2, centerX+size/2+head, centerY);
}

// biển ngưng
void drawStopSign(int cx, int cy, int radius) {
  // Tính toạ độ 8 đỉnh bát giác
  float angleStep = PI / 4; // 45 độ
  int x[8], y[8];
  for (int i = 0; i < 8; i++) {
    x[i] = cx + radius * cos(angleStep * i + PI/8); // xoay 22.5° để giống bát giác STOP
    y[i] = cy + radius * sin(angleStep * i + PI/8);
  }

  // Vẽ bát giác
  for (int i = 0; i < 8; i++) {
    int next = (i + 1) % 8;
    u8g2.drawLine(x[i], y[i], x[next], y[next]);
  }

  // Vẽ chữ "STOP" giữa
  u8g2.setFont(u8g2_font_ncenB14_tr); // font chữ to
  int tw = u8g2.getStrWidth("STOP");
  u8g2.drawStr(cx - tw/2, cy + 5, "STOP");
}

// Các hàm điều khiển

void goForward() {
  Serial.println("Forward");
  currentAction = "FORWARD";
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH); 
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW); 
  analogWrite(enable1Pin, dutyCycleLeft);
  analogWrite(enable2Pin, dutyCycleRight);
  // delay(3000);
  // ledcWrite(pwmChannel1, 200);
  // ledcWrite(pwmChannel2, 200);
  // digitalWrite(motor1Pin1, LOW);
  // digitalWrite(motor1Pin2, LOW);
  // digitalWrite(motor2Pin1, LOW);
  // digitalWrite(motor2Pin2, LOW);
}

void goBackward() {
  Serial.println("Back");
  currentAction = "BACK";
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW); 
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH);
  analogWrite(enable1Pin, dutyCycleLeft);
  analogWrite(enable2Pin, dutyCycleRight);
  // delay(3000); 
  // ledcWrite(pwmChannel1, 200);
  // ledcWrite(pwmChannel2, 200);
  // digitalWrite(motor1Pin1, LOW);
  // digitalWrite(motor1Pin2, LOW);
  // digitalWrite(motor2Pin1, LOW);
  // digitalWrite(motor2Pin2, LOW);
}

void stop() {
  Serial.println("Stop");
  currentAction = "STOP";
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, LOW);
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, LOW);
  analogWrite(enable1Pin, 0);
  analogWrite(enable2Pin, 0);
  // delay(5000);
  // ledcWrite(pwmChannel1, 0);
  // ledcWrite(pwmChannel2, 0);
}

void turnLeft() {
  Serial.println("Left");
  currentAction = "LEFT";
  digitalWrite(motor1Pin1, HIGH);
  digitalWrite(motor1Pin2, LOW); 
  digitalWrite(motor2Pin1, HIGH);
  digitalWrite(motor2Pin2, LOW); 
  analogWrite(enable1Pin, 10);
  analogWrite(enable2Pin, 150);
  // delay(3000);
  // ledcWrite(pwmChannel1, 0);
  // ledcWrite(pwmChannel2, 200);
  // digitalWrite(motor1Pin1, LOW);
  // digitalWrite(motor1Pin2, LOW);
  // digitalWrite(motor2Pin1, LOW);
  // digitalWrite(motor2Pin2, LOW);
}

void turnRight() {
  Serial.println("Right");
  currentAction = "RIGHT";
  digitalWrite(motor1Pin1, LOW);
  digitalWrite(motor1Pin2, HIGH); 
  digitalWrite(motor2Pin1, LOW);
  digitalWrite(motor2Pin2, HIGH); 
  analogWrite(enable1Pin, 150);
  analogWrite(enable2Pin, 10);
  // delay(3000);
  // ledcWrite(pwmChannel1, 200);
  // ledcWrite(pwmChannel2, 0);
  // digitalWrite(motor1Pin1, LOW);
  // digitalWrite(motor1Pin2, LOW);
  // digitalWrite(motor2Pin1, LOW);
  // digitalWrite(motor2Pin2, LOW);
}
