import os
import psutil
import subprocess
import time
import json

class CommandLibrary:
    def __init__(self):
        self.commands = {
            "CHECK_STATUS": self.check_status,
            "SHUTDOWN": self.shutdown,
            "RESTART": self.restart,
            "GET_TEMPERATURE": self.get_temperature,
            "GET_CPU_USAGE": self.get_cpu_usage,
            "GET_RAM_USAGE": self.get_ram_usage,
            "GET_I2C_DEVICES": self.get_i2c_devices,
        }

    def execute_command(self, command, *args):
        if command in self.commands:
            result = self.commands[command](*args)
            return json.dumps({"commande": command, "résultat": result}, indent=4)
        else:
            return json.dumps({"erreur": "Commande inconnue"}, indent=4)

    def check_status(self):
        temp = self.get_temperature()
        cpu_usage = self.get_cpu_usage()
        ram_usage = self.get_ram_usage()
        i2c_status = self.get_i2c_devices()
        return {
            "Température": temp,
            "CPU": f"{cpu_usage}%",
            "RAM": f"{ram_usage}%",
            "I2C": i2c_status
        }

    def shutdown(self):
        os.system("sudo shutdown -h now")
        return "Arrêt du système en cours..."
    
    def restart(self):
        os.system("sudo reboot")
        return "Redémarrage du système en cours..."
    
    def get_temperature(self):
        return os.popen("vcgencmd measure_temp").readline().strip()
    
    def get_cpu_usage(self):
        return psutil.cpu_percent()
    
    def get_ram_usage(self):
        return psutil.virtual_memory().percent
    
    def get_i2c_devices(self):
        try:
            i2c_devices = subprocess.check_output("i2cdetect -y 1", shell=True).decode()
            return "Capteurs I2C détectés" if "UU" in i2c_devices else "Aucun capteur I2C trouvé"
        except Exception as e:
            return f"Erreur de lecture I2C: {e}"

# Exemple
if __name__ == "__main__":
    command_lib = CommandLibrary()
    print(command_lib.execute_command("CHECK_STATUS"))
