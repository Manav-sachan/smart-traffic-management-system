// Define the LED Pins
const int redPin = 12;
const int greenPin = 11;

void setup() {
  // Start listening to the USB port at 9600 baud rate
  Serial.begin(9600);
  
  // Set pins as outputs
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  
  // Default State: Normal Traffic (Red Light)
  digitalWrite(redPin, HIGH);
  digitalWrite(greenPin, LOW);
}

void loop() {
  // Check if Python has sent any data down the USB cable
  if (Serial.available() > 0) {
    char command = Serial.read(); // Read the command

    if (command == 'E') {
      // 'E' for Emergency -> Immediate Green3
      
      digitalWrite(redPin, LOW);
      digitalWrite(greenPin, HIGH);
    } 
    else if (command == 'H') {
      // 'H' for High Density -> Extend Green
      digitalWrite(redPin, LOW);
      digitalWrite(greenPin, HIGH);
    } 
    else if (command == 'N') {
      // 'N' for Normal Traffic -> Red Light
    
      digitalWrite(redPin, HIGH);
      digitalWrite(greenPin, LOW);
    }
  }
}