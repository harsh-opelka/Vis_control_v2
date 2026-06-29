# from opcua import Client, ua
# import time
# import sys
#
# PLC_URL = "opc.tcp://192.168.224.100:4840"
# RECONNECT_DELAY = 2.0
# STATUS_POLL = 0.2
#
# NODES = {
#     "ext_tuchabzug_stop": "ns=6;s=::opcua:ext_tuchabzug_stop",
#     "ext_error": "ns=6;s=::opcua:ext_error",
#     "ext_error_quit": "ns=6;s=::opcua:ext_error_quit",
#     "ext_tuchabzug_status": "ns=6;s=::opcua:ext_tuchabzug_status",
# }
#
#
# class OpcUaController:
#     def __init__(self, url, nodes):
#         self.url = url
#         self.node_ids = nodes
#         self.client = None
#         self.nodes = {}
#         self.connected = False
#
#     def connect(self):
#         self.disconnect()
#         self.client = Client(self.url)
#         self.client.connect()
#         self.nodes = {name: self.client.get_node(node_id) for name, node_id in self.node_ids.items()}
#         self.connected = True
#
#     def disconnect(self):
#         if self.client is not None:
#             try:
#                 self.client.disconnect()
#             except Exception:
#                 pass
#         self.client = None
#         self.nodes = {}
#         self.connected = False
#
#     def ensure_connection(self):
#         while True:
#             try:
#                 if not self.connected or self.client is None:
#                     print(f"Verbinde mit PLC {self.url} ...")
#                     self.connect()
#                     print("Verbindung steht.")
#                 else:
#                     self.nodes["ext_tuchabzug_status"].get_value()
#                 return
#             except Exception as e:
#                 self.connected = False
#                 print(f"Verbindung fehlgeschlagen/verloren: {e}")
#                 print(f"Neuer Versuch in {RECONNECT_DELAY:.1f}s ...")
#                 time.sleep(RECONNECT_DELAY)
#
#     def read_bool(self, name):
#         self.ensure_connection()
#         try:
#             return bool(self.nodes[name].get_value())
#         except Exception:
#             self.connected = False
#             raise
#
#     def read_int(self, name):
#         self.ensure_connection()
#         try:
#             return int(self.nodes[name].get_value())
#         except Exception:
#             self.connected = False
#             raise
#
#     def write_bool(self, name, value):
#         self.ensure_connection()
#         try:
#             dv = ua.DataValue(ua.Variant(bool(value), ua.VariantType.Boolean))
#             self.nodes[name].set_value(dv)
#         except Exception:
#             self.connected = False
#             raise
#
#     def write_int16(self, name, value):
#         self.ensure_connection()
#         try:
#             ivalue = int(value)
#             if ivalue < -32768 or ivalue > 32767:
#                 raise ValueError("INT16 außerhalb des gültigen Bereichs (-32768..32767)")
#             dv = ua.DataValue(ua.Variant(ivalue, ua.VariantType.Int16))
#             self.nodes[name].set_value(dv)
#         except Exception:
#             self.connected = False
#             raise
#
#     def pulse_stop_if_status_true(self):
#         status = self.read_bool("ext_tuchabzug_status")
#         print(f"ext_tuchabzug_status = {status}")
#         if not status:
#             print("Kein Stop-Impuls gesendet, da Status FALSE ist.")
#             return
#         self.write_bool("ext_tuchabzug_stop", True)
#         print("ext_tuchabzug_stop = TRUE")
#         time.sleep(0.1)
#         self.write_bool("ext_tuchabzug_stop", False)
#         print("ext_tuchabzug_stop = FALSE")
#
#     def set_error_and_wait_for_quit(self, error_value=2):
#         self.write_int16("ext_error", error_value)
#         print(f"ext_error = {error_value}")
#         print("Warte auf ext_error_quit = TRUE ...")
#         while True:
#             try:
#                 quit_value = self.read_bool("ext_error_quit")
#                 print(f"ext_error_quit = {quit_value}")
#                 if quit_value:
#                     self.write_int16("ext_error", 0)
#                     print("ext_error = 0")
#                     return
#                 time.sleep(STATUS_POLL)
#             except Exception as e:
#                 print(f"Fehler beim Warten auf Quit: {e}")
#                 time.sleep(RECONNECT_DELAY)
#
#     def show_status(self):
#         status = self.read_bool("ext_tuchabzug_status")
#         error_quit = self.read_bool("ext_error_quit")
#         error_val = self.read_int("ext_error")
#         print("Aktuelle Werte:")
#         print(f"  ext_tuchabzug_status = {status}")
#         print(f"  ext_tuchabzug_stop   = <Write-Only per Skript, Wert nicht extra gelesen>")
#         print(f"  ext_error            = {error_val}")
#         print(f"  ext_error_quit       = {error_quit}")
#
#
# def print_menu():
#     print()
#     print("OPC UA Konsole")
#     print("1 - Verbindung prüfen")
#     print("2 - Statuswerte anzeigen")
#     print("3 - Stop-Impuls senden (nur wenn ext_tuchabzug_status = TRUE)")
#     print("4 - ext_error = 2 setzen und auf ext_error_quit warten")
#     print("5 - ext_error manuell setzen")
#     print("6 - ext_error auf 0 setzen")
#     print("7 - Neu verbinden")
#     print("q - Beenden")
#     print()
#
# def main():
#     controller = OpcUaController(PLC_URL, NODES)
#
#     try:
#         controller.ensure_connection()
#
#         while True:
#             print_menu()
#             choice = input("Auswahl: ").strip().lower()
#
#             try:
#                 if choice == "1":
#                     controller.ensure_connection()
#                     print("Verbindung ist OK.")
#                 elif choice == "2":
#                     controller.show_status()
#                 elif choice == "3":
#                     controller.pulse_stop_if_status_true()
#                 elif choice == "4":
#                     controller.set_error_and_wait_for_quit(2)
#                 elif choice == "5":
#                     value = input("Bitte INT-Wert für ext_error eingeben: ").strip()
#                     controller.write_int16("ext_error", int(value))
#                     print(f"ext_error = {int(value)}")
#                 elif choice == "6":
#                     controller.write_int16("ext_error", 0)
#                     print("ext_error = 0")
#                 elif choice == "7":
#                     controller.disconnect()
#                     controller.ensure_connection()
#                 elif choice == "q":
#                     print("Beende Programm.")
#                     break
#                 else:
#                     print("Ungültige Auswahl.")
#             except KeyboardInterrupt:
#                 print("\nAktion abgebrochen.")
#             except Exception as e:
#                 print(f"Fehler bei der Aktion: {e}")
#
#     finally:
#         controller.disconnect()
#
#
# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nProgramm durch Benutzer beendet.")
#         sys.exit(0)



from opcua import Client, ua
import time
import sys

PLC_URL = "opc.tcp://192.168.224.100:4840"
RECONNECT_DELAY = 2.0
STATUS_POLL = 0.2

NODES = {
    "ext_tuchabzug_stop": "ns=6;s=::TUA:fromext_stop_Tuchabzug",
    "ext_error": "ns=6;s=::TUA:fromext_Error_idx",
    "ext_error_quit": "ns=6;s=::TUA:toext_Error_quit",
    "ext_tuchabzug_status": "ns=6;s=::TUA:toext_Tuchabzug_running",
}


class OpcUaController:
    def __init__(self, url, nodes):
        self.url = url
        self.node_ids = nodes
        self.client = None
        self.nodes = {}
        self.connected = False

    def connect(self):
        self.disconnect()
        self.client = Client(self.url)
        self.client.connect()
        self.nodes = {name: self.client.get_node(node_id) for name, node_id in self.node_ids.items()}
        self.connected = True

    def disconnect(self):
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                pass
        self.client = None
        self.nodes = {}
        self.connected = False

    def ensure_connection(self):
        while True:
            try:
                if not self.connected or self.client is None:
                    print(f"Verbinde mit PLC {self.url} ...")
                    self.connect()
                    print("Verbindung steht.")
                else:
                    self.nodes["ext_tuchabzug_status"].get_value()
                return
            except Exception as e:
                self.connected = False
                print(f"Verbindung fehlgeschlagen/verloren: {e}")
                print(f"Neuer Versuch in {RECONNECT_DELAY:.1f}s ...")
                time.sleep(RECONNECT_DELAY)

    def read_bool(self, name):
        self.ensure_connection()
        try:
            return bool(self.nodes[name].get_value())
        except Exception:
            self.connected = False
            raise

    def read_int(self, name):
        self.ensure_connection()
        try:
            return int(self.nodes[name].get_value())
        except Exception:
            self.connected = False
            raise

    def write_bool(self, name, value):
        self.ensure_connection()
        try:
            dv = ua.DataValue(ua.Variant(bool(value), ua.VariantType.Boolean))
            self.nodes[name].set_value(dv)
        except Exception:
            self.connected = False
            raise

    def write_uint16(self, name, value):
        self.ensure_connection()
        try:
            ivalue = int(value)
            if ivalue < 0 or ivalue > 65535:
                raise ValueError("UINT16 außerhalb des gültigen Bereichs (0..65535)")
            dv = ua.DataValue(ua.Variant(ivalue, ua.VariantType.UInt16))
            self.nodes[name].set_value(dv)
        except Exception:
            self.connected = False
            raise

    def pulse_stop_if_status_true(self):
        status = self.read_bool("ext_tuchabzug_status")
        print(f"toext_Tuchabzug_running = {status}")
        if not status:
            print("Kein Stop-Impuls gesendet, da Status FALSE ist.")
            return
        self.write_bool("ext_tuchabzug_stop", True)
        print("fromext_stop_Tuchabzug = TRUE")
        time.sleep(0.1)
        self.write_bool("ext_tuchabzug_stop", False)
        print("fromext_stop_Tuchabzug = FALSE")

    def set_error_and_wait_for_quit(self, error_value=2):
        self.write_uint16("ext_error", error_value)
        print(f"fromext_Error_idx = {error_value}")
        print("Warte auf toext_Error_quit = TRUE ...")
        while True:
            try:
                quit_value = self.read_bool("ext_error_quit")
                print(f"toext_Error_quit = {quit_value}")
                if quit_value:
                    self.write_uint16("ext_error", 0)
                    print("fromext_Error_idx = 0 (zurückgesetzt)")
                    return
                time.sleep(STATUS_POLL)
            except Exception as e:
                print(f"Fehler beim Warten auf Quit: {e}")
                time.sleep(RECONNECT_DELAY)

    def show_status(self):
        status = self.read_bool("ext_tuchabzug_status")
        error_quit = self.read_bool("ext_error_quit")
        error_val = self.read_int("ext_error")
        print("Aktuelle Werte:")
        print(f"  toext_Tuchabzug_running  = {status}")
        print(f"  fromext_stop_Tuchabzug   = <Write-Only per Skript, Wert nicht extra gelesen>")
        print(f"  fromext_Error_idx        = {error_val}")
        print(f"  toext_Error_quit         = {error_quit}")


def print_menu():
    print()
    print("OPC UA Konsole")
    print("1 - Verbindung prüfen")
    print("2 - Statuswerte anzeigen")
    print("3 - Stop-Impuls senden (nur wenn toext_Tuchabzug_running = TRUE)")
    print("4 - fromext_Error_idx = 2 setzen und auf toext_Error_quit warten")
    print("5 - fromext_Error_idx manuell setzen")
    print("6 - fromext_Error_idx auf 0 setzen")
    print("7 - Neu verbinden")
    print("q - Beenden")
    print()


def main():
    controller = OpcUaController(PLC_URL, NODES)

    try:
        controller.ensure_connection()

        while True:
            print_menu()
            choice = input("Auswahl: ").strip().lower()

            try:
                if choice == "1":
                    controller.ensure_connection()
                    print("Verbindung ist OK.")
                elif choice == "2":
                    controller.show_status()
                elif choice == "3":
                    controller.pulse_stop_if_status_true()
                elif choice == "4":
                    controller.set_error_and_wait_for_quit(2)
                elif choice == "5":
                    value = input("Bitte UINT-Wert für fromext_Error_idx eingeben: ").strip()
                    controller.write_uint16("ext_error", int(value))
                    print(f"fromext_Error_idx = {int(value)}")
                elif choice == "6":
                    controller.write_uint16("ext_error", 0)
                    print("fromext_Error_idx = 0")
                elif choice == "7":
                    controller.disconnect()
                    controller.ensure_connection()
                elif choice == "q":
                    print("Beende Programm.")
                    break
                else:
                    print("Ungültige Auswahl.")
            except KeyboardInterrupt:
                print("\nAktion abgebrochen.")
            except Exception as e:
                print(f"Fehler bei der Aktion: {e}")

    finally:
        controller.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramm durch Benutzer beendet.")
        sys.exit(0)