#include <WiFi.h>
#include <ESP32Servo.h>

// ====== WiFi 设置 ======
const char* ssid = "TANK_MINI_1";
const char* password = "7892580aa";

// ====== 舵机设置 ======
Servo continuousServo;
const int servoPin = 21; // GP21

const int STOP_US = 1500;
const int FORWARD_US = 800;   // 正转（开）
const int BACKWARD_US = 2200; // 反转（关）

// 默认参数
int forwardDuration = 400;
int backwardDuration = 400;
int stopDuration = 300;

WiFiServer server(80);

void setup() {
  Serial.begin(115200);
  
  continuousServo.setPeriodHertz(50);
  continuousServo.attach(servoPin, 500, 2500);
  continuousServo.writeMicroseconds(STOP_US);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected!");
  Serial.println("IP: " + WiFi.localIP().toString());

  server.begin();
}

// 发送 HTML 页面
void handleRoot(WiFiClient& client) {
  String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>舵机开关控制</title>
  <style>
    body { 
      font-family: -apple-system, Arial; 
      text-align: center; 
      padding: 20px; 
      background: #f0f0f0; 
    }
    .btn { 
      padding: 18px 36px; 
      font-size: 20px; 
      margin: 15px; 
      border: none; 
      border-radius: 12px; 
      cursor: pointer;
    }
    .btn-on { background: #4CAF50; color: white; }
    .btn-off { background: #f44336; color: white; }
    .btn-setting { background: #2196F3; color: white; }
    .settings-panel {
      background: white;
      padding: 20px;
      border-radius: 10px;
      margin: 20px auto;
      max-width: 500px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .param-control {
      margin: 15px 0;
      text-align: left;
    }
    label {
      display: inline-block;
      width: 180px;
      text-align: right;
      margin-right: 10px;
    }
    input[type="range"] {
      width: 200px;
      vertical-align: middle;
    }
    input[type="number"] {
      width: 80px;
      padding: 5px;
      text-align: center;
    }
    #status { 
      margin-top: 20px; 
      font-size: 18px; 
      color: #333; 
      min-height: 27px;
    }
    .value-display {
      display: inline-block;
      width: 50px;
      text-align: left;
    }
  </style>
</head>
<body>
  <h2>💡 iPad 控制电灯</h2>
  
  <button class="btn btn-on" onclick="action('/on')">开灯</button>
  <button class="btn btn-off" onclick="action('/off')">关灯</button>
  
  <div class="settings-panel">
    <h3>⚙️ 参数设置</h3>
    <div class="param-control">
      <label for="forwardDuration">正转时间 (ms):</label>
      <input type="range" id="forwardDuration" min="100" max="2000" value="400" oninput="updateValue('forwardDuration')">
      <input type="number" id="forwardDurationVal" min="100" max="2000" value="400" onchange="syncSlider('forwardDuration')">
    </div>
    
    <div class="param-control">
      <label for="backwardDuration">反转时间 (ms):</label>
      <input type="range" id="backwardDuration" min="100" max="2000" value="400" oninput="updateValue('backwardDuration')">
      <input type="number" id="backwardDurationVal" min="100" max="2000" value="400" onchange="syncSlider('backwardDuration')">
    </div>
    
    <div class="param-control">
      <label for="stopDuration">停止时间 (ms):</label>
      <input type="range" id="stopDuration" min="100" max="1000" value="300" oninput="updateValue('stopDuration')">
      <input type="number" id="stopDurationVal" min="100" max="1000" value="300" onchange="syncSlider('stopDuration')">
    </div>
    
    <button class="btn btn-setting" onclick="saveSettings()">保存设置</button>
  </div>
  
  <p id="status">就绪</p>
  
  <script>
    // 更新数值显示
    function updateValue(paramId) {
      const slider = document.getElementById(paramId);
      const numberInput = document.getElementById(paramId + 'Val');
      numberInput.value = slider.value;
    }
    
    // 同步滑块和数字输入
    function syncSlider(paramId) {
      const slider = document.getElementById(paramId);
      const numberInput = document.getElementById(paramId + 'Val');
      slider.value = numberInput.value;
    }
    
    // 执行动作
    async function action(url) {
      const status = document.getElementById('status');
      status.innerText = '执行中...';
      try {
        await fetch(url);
        status.innerText = '✅ 完成！';
      } catch(e) {
        status.innerText = '❌ 失败';
      }
    }
    
    // 保存设置
    async function saveSettings() {
      const status = document.getElementById('status');
      const params = new URLSearchParams();
      params.append('forward', document.getElementById('forwardDuration').value);
      params.append('backward', document.getElementById('backwardDuration').value);
      params.append('stop', document.getElementById('stopDuration').value);
      
      status.innerText = '保存中...';
      try {
        await fetch('/settings?' + params.toString());
        status.innerText = '✅ 设置已保存！';
      } catch(e) {
        status.innerText = '❌ 保存失败';
      }
    }
    
    // 页面加载时获取当前设置
    window.onload = async function() {
      try {
        const response = await fetch('/get_settings');
        const settings = await response.json();
        document.getElementById('forwardDuration').value = settings.forward;
        document.getElementById('forwardDurationVal').value = settings.forward;
        document.getElementById('backwardDuration').value = settings.backward;
        document.getElementById('backwardDurationVal').value = settings.backward;
        document.getElementById('stopDuration').value = settings.stop;
        document.getElementById('stopDurationVal').value = settings.stop;
      } catch(e) {
        console.log('无法获取设置');
      }
    }
  </script>
</body>
</html>
)rawliteral";

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html; charset=utf-8");
  client.println("Connection: close");
  client.println();
  client.print(html);
}

// 发送简单响应（如 /on /off 完成后）
void sendOK(WiFiClient& client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/plain; charset=utf-8");
  client.println("Connection: close");
  client.println();
  client.print("操作完成");
}

// 发送JSON格式的设置
void sendSettings(WiFiClient& client) {
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: application/json");
  client.println("Connection: close");
  client.println();
  client.printf("{\"forward\":%d,\"backward\":%d,\"stop\":%d}", 
                forwardDuration, backwardDuration, stopDuration);
}

void turnOn() {
  continuousServo.writeMicroseconds(FORWARD_US);
  delay(forwardDuration);
  continuousServo.writeMicroseconds(STOP_US);
  delay(stopDuration);
}

void turnOff() {
  continuousServo.writeMicroseconds(BACKWARD_US);
  delay(backwardDuration);
  continuousServo.writeMicroseconds(STOP_US);
  delay(stopDuration);
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    String req = client.readStringUntil('\r');
    client.flush();

    if (req.indexOf("GET / ") != -1) {
      handleRoot(client);
    } else if (req.indexOf("GET /on") != -1) {
      turnOn();
      sendOK(client);
    } else if (req.indexOf("GET /off") != -1) {
      turnOff();
      sendOK(client);
    } else if (req.indexOf("GET /settings") != -1) {
      // 解析参数并更新设置
      if (req.indexOf("?") != -1) {
        String params = req.substring(req.indexOf("?") + 1);
        int pos1 = params.indexOf("forward=");
        int pos2 = params.indexOf("&", pos1);
        if (pos1 != -1 && pos2 != -1) {
          forwardDuration = params.substring(pos1 + 8, pos2).toInt();
        }
        
        pos1 = params.indexOf("backward=");
        pos2 = params.indexOf("&", pos1);
        if (pos1 != -1 && pos2 != -1) {
          backwardDuration = params.substring(pos1 + 9, pos2).toInt();
        } else if (pos1 != -1) {
          backwardDuration = params.substring(pos1 + 9).toInt();
        }
        
        pos1 = params.indexOf("stop=");
        if (pos1 != -1) {
          stopDuration = params.substring(pos1 + 5).toInt();
        }
      }
      sendOK(client);
    } else if (req.indexOf("GET /get_settings") != -1) {
      sendSettings(client);
    } else {
      // 404
      client.println("HTTP/1.1 404 Not Found");
      client.println("Content-Type: text/plain; charset=utf-8");
      client.println("Connection: close");
      client.println();
      client.print("页面未找到");
    }

    // 确保数据发送完毕
    delay(1);
    client.stop();
  }
}