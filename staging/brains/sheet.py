"""Tile a sweep's thumbnails into one contact sheet, numbered, so the search
can be judged by eye and not only by its own score."""
import json, sys
import numpy as np
from PIL import Image, ImageDraw

path = sys.argv[1]
order = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
rows = json.load(open(path))
thumbs = np.load(path.replace(".json", "_thumbs.npy"))
idx = order if order else list(range(len(rows)))
cols = int(np.ceil(np.sqrt(len(idx))))
n = len(idx)
rowsn = int(np.ceil(n / cols))
T, PADT = thumbs.shape[1], 14
sheet = Image.new("RGB", (cols * (T + 4), rowsn * (T + PADT + 4)), (10, 12, 16))
dr = ImageDraw.Draw(sheet)
for j, i in enumerate(idx):
    x, y = (j % cols) * (T + 4) + 2, (j // cols) * (T + PADT + 4) + 2
    sheet.paste(Image.fromarray(thumbs[i]).transpose(Image.FLIP_TOP_BOTTOM), (x, y))
    dr.text((x + 1, y + T + 1), f"{i}", fill=(190, 200, 215))
sheet = sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST)
sheet.save(sys.argv[3] if len(sys.argv) > 3 else "sheet.png")
print(f"{n} tiles, {cols}x{rowsn}, {sheet.width}x{sheet.height}")
