#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <TM1637Display.h>
#include <ArduinoJson.h>

// WiFi credentials
#define WIFI_SSID "demo"
#define WIFI_PASSWORD "demo1234"

// Two API endpoints
// 1) Returns {"Birds":0,"Persons":0}
#define DETECTION_API_URL "http://demo/detection_data"

// 2) Returns {"temperature":26.13,"door":"Open"}
#define DOOR_TEMP_API_URL "http://demo/data"

// TM1637 pins
#define CLK_PIN D1  // Clock
#define DIO_PIN D2  // Data

TM1637Display display(CLK_PIN, DIO_PIN);

void setup() {
  Serial.begin(115200);
  delay(10);
  Serial.println("\nConnecting to WiFi...");

  // Connect to WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  // Simple loading animation on the display while connecting
  int animPos = 0;
  while (WiFi.status() != WL_CONNECTED) {
    uint8_t segments[4] = {0, 0, 0, 0};
    segments[animPos] = 0x40;  // Dash indicator
    display.setSegments(segments);
    animPos = (animPos + 1) % 4;
    delay(200);
    Serial.print(".");
  }
  
  Serial.println("\nConnected! IP address: " + WiFi.localIP().toString());
  
  display.setBrightness(0x0f);  // Maximum brightness
  display.clear();
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // --- 1) Get detection data (Birds, Persons) ---
    int persons = 0;
    int birds = 0;
    {
      HTTPClient http;
      WiFiClient client;
      http.begin(client, DETECTION_API_URL);
      int httpCode = http.GET();
      if (httpCode > 0) {
        String payload = http.getString();
        Serial.println("Detection payload: " + payload);
        
        DynamicJsonDocument doc(256);
        DeserializationError error = deserializeJson(doc, payload);
        if (!error) {
          persons = doc["Persons"];
          birds   = doc["Birds"];
          Serial.print("Parsed Persons: ");
          Serial.println(persons);
          Serial.print("Parsed Birds: ");
          Serial.println(birds);
        } else {
          Serial.println("JSON parsing failed for detection_data: " + String(error.c_str()));
        }
      } else {
        Serial.print("Detection HTTP error: ");
        Serial.println(httpCode);
      }
      http.end();
    }

    // --- 2) Get door & temp data ---
    float temperature = 0.0;
    String doorState = "Unknown";
    {
      HTTPClient http;
      WiFiClient client;
      http.begin(client, DOOR_TEMP_API_URL);
      int httpCode = http.GET();
      if (httpCode > 0) {
        String payload = http.getString();
        Serial.println("Door/Temp payload: " + payload);
        
        DynamicJsonDocument doc(256);
        DeserializationError error = deserializeJson(doc, payload);
        if (!error) {
          temperature = doc["temperature"];
          doorState   = doc["door"].as<String>();
          Serial.print("Parsed Temperature: ");
          Serial.println(temperature);
          Serial.print("Parsed Door: ");
          Serial.println(doorState);
        } else {
          Serial.println("JSON parsing failed for door/temp data: " + String(error.c_str()));
        }
      } else {
        Serial.print("Door/Temp HTTP error: ");
        Serial.println(httpCode);
      }
      http.end();
    }

    // --- Prepare to display data ---
    // 1st digit: Persons (0..9)
    // 2nd digit: Birds (0..9)
    // 3rd & 4th digits: Temperature (integer, 0..99)
    
    if (persons < 0) persons = 0;
    if (persons > 9) persons = 9;
    if (birds < 0)   birds = 0;
    if (birds > 9)   birds = 9;

    // Round temperature to nearest int
    int tempInt = (int) round(temperature);
    if (tempInt < 0)  tempInt = 0;   // clamp negative
    if (tempInt > 99) tempInt = 99;  // clamp above 99

    int tens = tempInt / 10;
    int ones = tempInt % 10;

    // Build the 4-digit array
    uint8_t seg[4];
    seg[0] = display.encodeDigit(persons);
    seg[1] = display.encodeDigit(birds);
    seg[2] = display.encodeDigit(tens);
    seg[3] = display.encodeDigit(ones);
    display.setSegments(seg);

  } else {
    Serial.println("WiFi not connected");
  }
  
  // Update every 10 seconds
  delay(10000);
}
