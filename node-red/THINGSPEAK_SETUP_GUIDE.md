# ThingSpeak Setup Guide - 3 Fields Configuration

## 🔧 **ThingSpeak Channel Setup**

### **Step 1: Tạo Channel mới**
1. Truy cập: https://thingspeak.com/
2. Đăng ký/Đăng nhập tài khoản
3. Click **"Channels"** → **"New Channel"**

### **Step 2: Cấu hình Channel**
```
Channel Name: Hand Gesture Car Error Monitoring
Description: Error logging system for hand gesture recognition car control
```

### **Step 3: Cấu hình 3 Fields**

#### **Field 1: Severity Level (Mức độ nghiêm trọng)**
```
Field 1: Severity Level
Description: Error severity (1=Low, 2=Medium, 3=High)
```

#### **Field 2: Error Message (Thông báo lỗi)**
```
Field 2: Error Message  
Description: Combined error message with type and details
```

#### **Field 3: Timestamp (Thời gian)**
```
Field 3: Timestamp
Description: ISO timestamp when error occurred
```

### **Step 4: Các Field khác**
- **Field 4, 5, 6, 7, 8:** Để trống (không sử dụng)

### **Step 5: Channel Settings**
```
✅ Make Channel Public (nếu muốn chia sẻ)
✅ Enable Channel View
✅ Show Channel Location: No
```

### **Step 6: Lấy API Key**
1. Sau khi tạo channel, vào tab **"API Keys"**
2. Copy **"Write API Key"** 
3. Cập nhật trong Node-RED:

```javascript
// Trong node "ThingSpeak Formatter"
api_key: 'YOUR_WRITE_API_KEY_HERE'  // Thay bằng API key của bạn
```

## 📊 **Data Structure**

### **Field Mapping:**
```
Field 1 (Severity Level):
- 1 = Low (Info/Debug)
- 2 = Medium (Warning) 
- 3 = High (Error/Critical)

Field 2 (Error Message):
- Format: "[TYPE] Message (from: source)"
- Example: "[SYSTEM] Hand gesture recognition system stopped"
- Example: "[CAMERA] Camera disconnected - System shutdown"

Field 3 (Timestamp):
- Format: ISO 8601 (2025-08-02T10:30:45.123Z)
- Automatic từ Node-RED
```

## 🔍 **Visualization Setup**

### **Chart 1: Severity Level Over Time**
```
Chart Type: Line Chart
X-Axis: created_at (time)
Y-Axis: field1 (Severity Level)
Title: "Error Severity Trends"
```

### **Chart 2: Error Count by Severity**
```
Chart Type: Bar Chart/Gauge
Data: Count of field1 values
Title: "Error Distribution"
Options:
- 1 (Low): Green
- 2 (Medium): Orange  
- 3 (High): Red
```

### **Table View: Recent Errors**
```
Display: Latest 20 entries
Columns: 
- Entry ID
- Severity Level (field1)
- Message (field2) 
- Timestamp (field3)
- Created At
```

## ⚡ **Rate Limiting**
```
ThingSpeak Free Limits:
- 3 million messages/year
- 8,200 messages/day
- 4 messages/minute average

Node-RED Rate Limiter:
- Thiết lập: 15 giây/message
- = 4 messages/minute = 5,760 messages/day
- ✅ An toàn trong giới hạn free
```

## 🧪 **Testing**

### **Test 1: Manual Error**
1. Trong Node-RED dashboard, tạo invalid command
2. Kiểm tra ThingSpeak channel
3. Verify data: field1=2, field2="[COMMAND] Invalid...", field3=timestamp

### **Test 2: System Shutdown**
1. Chạy `python app.py --enable_mqtt`
2. Tắt app (Ctrl+C)
3. Kiểm tra ThingSpeak: severity=3, message có "stopped"

### **Test 3: Rate Limiting**
1. Gửi nhiều errors nhanh liên tiếp
2. Verify chỉ 1 message/15 giây được gửi

## 📱 **Mobile/Web Access**

### **Public Channel URL:**
```
https://thingspeak.com/channels/YOUR_CHANNEL_ID
```

### **JSON API (để đọc data):**
```
https://api.thingspeak.com/channels/YOUR_CHANNEL_ID/feeds.json
https://api.thingspeak.com/channels/YOUR_CHANNEL_ID/feeds.json?results=10
```

### **Embed Charts:**
```html
<iframe src="https://thingspeak.com/channels/YOUR_CHANNEL_ID/charts/1?bgcolor=%23ffffff&color=%23d62020&dynamic=true&results=60&title=Error+Severity&type=line"></iframe>
```

## 🔧 **Node-RED Configuration**

### **Cập nhật API Key:**
1. Mở Node-RED editor
2. Double-click node **"ThingSpeak Formatter"**
3. Thay đổi dòng:
```javascript
api_key: 'LRGXV5Y7KITIE1V7'  // ← Thay bằng API key của bạn
```
4. Click **Deploy**

### **Test Connection:**
1. Deploy flow
2. Tạo error từ dashboard
3. Check Node-RED debug: "Sending to ThingSpeak (3 fields)"
4. Verify data xuất hiện trong ThingSpeak channel

## 📈 **Analytics Ideas**

### **Useful Queries:**
1. **Error rate per hour:** Group by hour, count entries
2. **Most common errors:** Group by message pattern
3. **System uptime:** Time between "started" and "stopped" 
4. **Error severity distribution:** Percentage of each severity level

### **Alerts Setup:**
1. **ThingAnalyze (paid):** React when field1 >= 3
2. **IFTTT integration:** Send email/SMS on high severity
3. **Custom webhooks:** Trigger notifications

## ✅ **Checklist**
- [ ] Tạo ThingSpeak channel với 3 fields
- [ ] Copy Write API Key
- [ ] Cập nhật API key trong Node-RED
- [ ] Deploy Node-RED flow
- [ ] Test bằng cách tạo error
- [ ] Verify data trong ThingSpeak
- [ ] Setup charts/visualizations
- [ ] Test rate limiting

## 🆘 **Troubleshooting**

### **Error: "API key invalid"**
- Kiểm tra API key đúng (Write API Key, không phải Read)
- Đảm bảo không có space thừa

### **Error: "Field value too long"**
- Message quá dài (>255 chars) 
- Node-RED đã tự cắt với `.substring(0, 255)`

### **Rate limiting issues:**
- ThingSpeak rate limit: check status code 429
- Node-RED rate limiter: check node status "Rate limited"

### **No data appears:**
- Check Node-RED debug logs
- Verify internet connection
- Check ThingSpeak channel status
