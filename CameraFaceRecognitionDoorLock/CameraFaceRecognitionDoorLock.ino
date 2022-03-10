//Viral Science www.youtube.com/c/viralscience  www.viralsciencecreativity.com
//ESP Camera Artificial Intelligence Face Detection Automatic Door Lock
#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "String.h"
//#include <Array.h>
#include <SPIFFS.h>
#include <FS.h>
#include "driver/rtc_io.h"
#include <EEPROM.h>            // read and write from flash memory
//#define FILE_PHOTO "/photo.jpg"

//
// WARNING!!! Make sure that you have either selected ESP32 Wrover Module,
//            or another board which has PSRAM enabled
//

// Pin definition for CAMERA_MODEL_AI_THINKER
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Select camera model
//#define CAMERA_MODEL_WROVER_KIT
//#define CAMERA_MODEL_ESP_EYE
//#define CAMERA_MODEL_M5STACK_PSRAM
//#define CAMERA_MODEL_M5STACK_WIDE
#define CAMERA_MODEL_AI_THINKER
#define Relay 2
#define sensor 4

#include "camera_pins.h"

WiFiClient espClient;
PubSubClient client(espClient);
const char* ssid = "Dinh Vung 3"; //Wifi Name SSID
const char* password = "19951995"; //WIFI Password

//MQTT config
#define PORT_MQTT 1883
const char* broker = "192.168.0.101";
const char* mqtt_client_name = "MQTT";
const char* mqtt_user = "vdquang";
const char* mqtt_pass = "1234";
const char* topic_lock = "doorlock/open";
const char* topic_capture = "doorlock/capture";



void startCameraServer();

boolean matchFace = false;
boolean activateRelay = false;
boolean appOpen = false;
boolean doorOpen = 0;
boolean test;
long prevMillis=0;
long startTime=0;
int interval = 5000;
long sensorTime = millis() - 61000;



void setup() {
  pinMode(Relay,OUTPUT);
  pinMode(sensor, INPUT);
  digitalWrite(Relay,LOW);
  
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();
 

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  //init with high specs to pre-allocate larger buffers
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }



   sensor_t * s = esp_camera_sensor_get();
  //initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);//flip it back
    s->set_brightness(s, 1);//up the blightness just a bit
    s->set_saturation(s, -2);//lower the saturation
  }
  //drop down frame size for higher initial frame rate
  s->set_framesize(s, FRAMESIZE_QVGA);

#if defined(CAMERA_MODEL_M5STACK_WIDE)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  client.setServer(broker, PORT_MQTT);

  startCameraServer();

  Serial.print("Camera Ready! Use 'http://");
  Serial.print(WiFi.localIP());
  Serial.println("' to connect");
  

  client.setCallback(callback);
 

    
}

void mqttESSP32()
{
 while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Thực hiện kết nối với mqtt user và pass
    if (client.connect(mqtt_client_name,mqtt_user, mqtt_pass)) {
      Serial.println("connected");
      // Khi kết nối sẽ publish thông báo
      client.publish(topic_lock, "ESP_reconnected");
      // ... và nhận lại thông tin này
      client.subscribe(topic_lock);
      client.subscribe(topic_capture);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Đợi 5s
      delay(5000);
    }
  }

  
}

void callback(char* topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  Serial.write(message, length);
  Serial.println(); 
  String doorStatus;
  
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    doorStatus += (char)message[i];
  }
  Serial.println();

  // Feel free to add more if statements to control more GPIOs with MQTT

  // If a message is received on the topic esp32/output, you check if the message is either "on" or "off". 
  // Changes the output state according to the message
 
  if (String(topic) == "doorlock/open") {
//    Serial.print("Open the door ");
    if(doorStatus == "open"){
      doorOpen = 1;
      Serial.print(doorOpen);
    }else {
      
      Serial.print(doorOpen);
    }
   }
 }
void loop() {

  if(!client.connected()) {
    mqttESSP32();
  }
  client.loop();
  
  test = digitalRead(sensor);


  if(test==0 && activateRelay==false) { 
   
    String sendingStr = "start";
    client.publish(topic_lock, sendingStr.c_str());
    client.loop(); // ham xu ly cua thu vien MQTT
    delay(11000);

    if(doorOpen == 1) {
    Serial.print("Open the door");
    activateRelay=true;
    digitalWrite(Relay,HIGH);
    prevMillis=millis();
    sensorTime = millis();

    }
   }

    
    if (activateRelay == true && millis()-prevMillis > interval)
    {
      activateRelay=false;
      digitalWrite(Relay,LOW);
    }

    if(doorOpen == 1 && appOpen == false) {
    doorOpen = 0;
    appOpen = true;
    Serial.print("Hello");
    digitalWrite(Relay, HIGH);
    startTime=millis();
    }  
    if(appOpen == true && millis()- startTime > interval) {
      appOpen = false;
      Serial.print("OK");
      digitalWrite(Relay, LOW);
    }
    
}
