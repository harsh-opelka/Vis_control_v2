# from opcua import Client, ua
# import time
#
# PLC_URL = "opc.tcp://192.168.224.100:4840"
#
# client = Client(PLC_URL)
# client.session_timeout = 30000
# client.secure_channel_timeout = 30000
#
# try:
#     client.connect()
#     print("CONNECTED!")
#     node = client.get_node("ns=6;s=::TUA:toext_Tuchabzug_running")
#     print(f"status = {node.get_value()}")
#     client.disconnect()
# except Exception as e:
#     print(f"Failed: {e}")
#     import traceback
#     traceback.print_exc()

# to find the node ids
from opcua import Client

PLC_URL = "opc.tcp://192.168.224.100:4840"

# Sections Tim mentioned in UA Expert. Add more if you see others.
SECTIONS = ["TUA", "Einlauf", "Signal", "Global", "PV"]


def browse_recursive(node, depth=0, max_depth=8, seen=None):
    if seen is None:
        seen = set()
    try:
        children = node.get_children()
    except Exception:
        return
    for child in children:
        try:
            node_id = child.nodeid.to_string()
            if node_id in seen:
                continue
            seen.add(node_id)
            try:
                browse_name = child.get_browse_name().Name
            except Exception:
                browse_name = "?"
            try:
                node_class = child.get_node_class().name
            except Exception:
                node_class = "?"
            # Print anything that is a Variable, or matches one of our sections of interest
            is_interesting = node_class == "Variable" or any(s in node_id for s in SECTIONS)
            # Skip the sub-property nodes like #TrueState / #EURange to reduce noise
            is_property = "#" in node_id
            if is_interesting and not is_property:
                print(f"{'  ' * depth}[{node_class:8}] {node_id}   ({browse_name})")
        except Exception:
            pass
        if depth < max_depth:
            browse_recursive(child, depth + 1, max_depth, seen)


def main():
    client = Client(PLC_URL)
    print(f"Verbinde mit {PLC_URL} ...")
    client.connect()
    print("Verbunden. Kompletter Adressraum (alle Sektionen):\n")
    try:
        objects = client.get_objects_node()
        browse_recursive(objects)
    finally:
        client.disconnect()
        print("\nFertig. Verbindung getrennt.")


if __name__ == "__main__":
    main()