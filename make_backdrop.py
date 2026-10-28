"""Write assets/backdrop.png: the film's background, a soft vignette under a faint grid of dots.

Standard library only: python3 make_backdrop.py
"""

import struct
import zlib
from pathlib import Path

W, H = 1920, 1080
UNIT = H / 8  # pixels per scene unit
SPACING = 0.5 * UNIT
DOT = 0.02 * UNIT

CENTER = (0x10, 0x17, 0x2E)  # the middle of the frame
EDGE = (0x07, 0x0A, 0x16)  # the corners
SPECK = (0x22, 0x2C, 0x4A)


def mix(a, b, t):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def main() -> None:
    rows = []
    cx, cy = W / 2, H / 2
    reach = (cx**2 + cy**2) ** 0.5
    for y in range(H):
        row = bytearray([0])  # filter: none
        gy = (y - cy) % SPACING
        dy = min(gy, SPACING - gy)
        for x in range(W):
            d = ((x - cx) ** 2 + ((y - cy) * 1.35) ** 2) ** 0.5 / reach
            color = mix(CENTER, EDGE, min(1.0, d) ** 1.6)
            gx = (x - cx) % SPACING
            dx = min(gx, SPACING - gx)
            r = (dx * dx + dy * dy) ** 0.5
            if r < DOT + 1:
                color = mix(color, SPECK, max(0.0, min(1.0, DOT + 0.5 - r)))
            row += bytes(color)
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"".join(rows), 9))
    png += chunk(b"IEND", b"")
    out = Path(__file__).resolve().parent / "assets" / "backdrop.png"
    out.write_bytes(png)
    print(out)


if __name__ == "__main__":
    main()
