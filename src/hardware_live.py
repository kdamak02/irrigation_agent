import serial
import time


class ESP32Connection:
    def __init__(self, port="COM5", baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None

    def connect(self):
        if self.ser is None or not self.ser.is_open:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)

    def parse_line(self, line):
        """
        Format attendu:
        MOISTURE:45;WATER:80
        """
        if not line.startswith("MOISTURE:"):
            return None, None

        try:
            parts = line.split(";")
            moisture = float(parts[0].split(":")[1])
            water = float(parts[1].split(":")[1])
            return moisture, water
        except Exception:
            return None, None

    def read_sensors(self):
        if self.ser is None or not self.ser.is_open:
            return None, None

        line = self.ser.readline().decode(errors="ignore").strip()
        return self.parse_line(line)

    def send_command(self, command):
        if self.ser is None or not self.ser.is_open:
            return

        self.ser.write((command + "\n").encode())

    def send_decision(self, action):
        if action == "PUMP_ON":
            self.send_command("ON")
        else:
            self.send_command("OFF")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()