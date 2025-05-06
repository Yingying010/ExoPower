#define SAMPLE_RATE 500  // 采样率
#define BAUD_RATE 115200 // 串口波特率
#define INPUT_PIN A0     // EMG 信号输入引脚

void setup() {
  Serial.begin(BAUD_RATE);
}

void loop() {
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;
  static long timer = 0;
  timer -= interval;

  if (timer < 0) {
    timer += 1000000 / SAMPLE_RATE;

    float raw_value = analogRead(INPUT_PIN);
    float filtered_value = Filter(raw_value);

    unsigned long timestamp_ms = millis(); // 获取当前时间（ms）
    
    // ✅ 输出格式为：时间戳,信号
    Serial.print(timestamp_ms);
    Serial.print(",");
    Serial.println(filtered_value);
  }
}

// Butterworth Bandpass Filter
float Filter(float input) { 
  float output = input;

  { static float z1, z2;
    float x = output - (-0.73945727 * z1) - (0.59923508 * z2);
    output = 0.00223489 * x + (0.00446978 * z1) + (0.00223489 * z2);
    z2 = z1; z1 = x;
  }
  { static float z1, z2;
    float x = output - (-1.03789224 * z1) - (0.64082390 * z2);
    output = 1.00000000 * x + (2.00000000 * z1) + (1.00000000 * z2);
    z2 = z1; z1 = x;
  }
  { static float z1, z2;
    float x = output - (-0.59186255 * z1) - (0.80647974 * z2);
    output = 1.00000000 * x + (-2.00000000 * z1) + (1.00000000 * z2);
    z2 = z1; z1 = x;
  }
  { static float z1, z2;
    float x = output - (-1.33318587 * z1) - (0.85392964 * z2);
    output = 1.00000000 * x + (-2.00000000 * z1) + (1.00000000 * z2);
    z2 = z1; z1 = x;
  }

  return output;
}
