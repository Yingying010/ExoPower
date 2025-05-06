#define SAMPLE_RATE 500  // 采样率
#define BAUD_RATE 115200 // 串口波特率

//官网：www.sichiray.com
//淘宝：http://brainlab.taobao.com
//淘宝店铺名称：大脑实验室

// 引脚
#define INPUT_PIN A0 // 信号输入

void setup()
{
  // 初始化串口
  Serial.begin(BAUD_RATE);
}

void loop()
{
  // 计算经过的时间
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;
  static long timer = 0;
  timer -= interval;

  // 采样
  if (timer < 0)
  {
    timer += 1000000 / SAMPLE_RATE;
    float sensor_value = analogRead(INPUT_PIN);
    float signal = Filter(sensor_value);
    Serial.println(signal);
  }
}
//官网：www.sichiray.com
//淘宝：http://brainlab.taobao.com
//淘宝店铺名称：大脑实验室
/*****************************由软件生成************************************/
//>>> Butterworth IIR Digital Filter: bandpass
//   Sampling Rate:500.0 Hz ,Frequency:[60.0, 100.0] Hz
//  Order: 4.0 ,implemented as second-order sections (biquads)
float Filter(float input)
{ 
  float output = input;
    {
        static float z1, z2; // filter section state
        float x = output - (-0.73945727*z1 )- (0.59923508*z2);
        output = 0.00223489*x + (0.00446978*z1 )+ (0.00223489*z2);
        z2 = z1;
        z1 = x;
    }
    
    {
        static float z1, z2; // filter section state
        float x = output - (-1.03789224*z1 )- (0.64082390*z2);
        output = 1.00000000*x + (2.00000000*z1 )+ (1.00000000*z2);
        z2 = z1;
        z1 = x;
    }
    
    {
        static float z1, z2; // filter section state
        float x = output - (-0.59186255*z1 )- (0.80647974*z2);
        output = 1.00000000*x + (-2.00000000*z1 )+ (1.00000000*z2);
        z2 = z1;
        z1 = x;
    }
    
    {
        static float z1, z2; // filter section state
        float x = output - (-1.33318587*z1 )- (0.85392964*z2);
        output = 1.00000000*x + (-2.00000000*z1 )+ (1.00000000*z2);
        z2 = z1;
        z1 = x;
    }
    
  return output;
}
//官网：www.sichiray.com
//淘宝：http://brainlab.taobao.com
//淘宝店铺名称：大脑实验室
