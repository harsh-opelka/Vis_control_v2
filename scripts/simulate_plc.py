"""Stand-in B&R PLC for development.

Connects to the VisControl OPC UA server and flips ``TuchabzugRunning`` between
True and False on a configurable cadence so the rest of the app sees pulse
events without a real PLC on the bench.

Usage:
    py scripts/simulate_plc.py --endpoint opc.tcp://127.0.0.1:4840/viscontrol/ \\
        --pulse-on 3.0 --pulse-off 4.0 --count 5

Defaults match ``config/default.yaml``.
"""

from __future__ import annotations

import argparse
import asyncio
import signal


async def _run(endpoint: str, pulse_on: float, pulse_off: float, count: int) -> int:
    try:
        from asyncua import Client  # type: ignore
    except ImportError:
        print("asyncua not installed.")
        return 1

    print(f"[sim] connecting to {endpoint}")
    async with Client(url=endpoint) as client:
        # Find the TuchabzugRunning node by browse path.
        objects = client.nodes.objects
        try:
            children = await objects.get_children()
            vis_folder = None
            for child in children:
                name = (await child.read_browse_name()).Name
                if name == "VisControl":
                    vis_folder = child
                    break
            if vis_folder is None:
                print("[sim] VisControl folder not found on server.")
                return 2
            tuch_var = None
            for child in await vis_folder.get_children():
                name = (await child.read_browse_name()).Name
                if name == "TuchabzugRunning":
                    tuch_var = child
                    break
            if tuch_var is None:
                print("[sim] TuchabzugRunning variable not found.")
                return 3
        except Exception as e:  # noqa: BLE001
            print(f"[sim] browse failed: {e}")
            return 4

        pulses = 0
        try:
            while count == 0 or pulses < count:
                print(f"[sim] pulse {pulses + 1}: TuchabzugRunning -> True")
                await tuch_var.write_value(True)
                await asyncio.sleep(pulse_on)
                print(f"[sim] pulse {pulses + 1}: TuchabzugRunning -> False")
                await tuch_var.write_value(False)
                await asyncio.sleep(pulse_off)
                pulses += 1
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("[sim] interrupted")
        finally:
            try:
                await tuch_var.write_value(False)
            except Exception:  # noqa: BLE001
                pass
        print(f"[sim] done after {pulses} pulses")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--endpoint", default="opc.tcp://127.0.0.1:4840/viscontrol/",
        help="VisControl OPC UA endpoint",
    )
    ap.add_argument("--pulse-on", type=float, default=3.0,
                    help="seconds TuchabzugRunning is held True")
    ap.add_argument("--pulse-off", type=float, default=4.0,
                    help="seconds TuchabzugRunning is held False")
    ap.add_argument("--count", type=int, default=0,
                    help="number of pulses (0 = run until Ctrl-C)")
    args = ap.parse_args()

    # On Windows asyncio needs a Selector loop policy for clean Ctrl-C with asyncua.
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except AttributeError:
        pass

    try:
        return asyncio.run(_run(args.endpoint, args.pulse_on, args.pulse_off, args.count))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
