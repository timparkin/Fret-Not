# make_fretboard.py
#from fretboard_labels import add_label_grid_onto_image, mode_colors

## REQUIRED libraries
## colorsys, PIL


import yaml
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageChops
import colorsys
from typing import List, Tuple, Union, Optional


MAKE = 'changes'



# Build labels:
rows, cols = 6, 15  # original grid (function will add the extra-left col)
total_cols = cols + 1

labels = []

# 2) A few ad-hoc grid-indexed overrides (row/col), zero-based


scale = [0, 2, 4, 5, 7, 9, 11, 12]
scale_step = [2, 2, 1, 2, 2, 2, 1]
string_step = [5, 4, 5, 5, 5, 5]

tone = ['1', 'b2', '2', 'b3', '3', '4', '#4', '5', 'b6', '6', 'b7', '7', ]



notes_mapping = {
    '1': 'C',
    'b2': 'C#',
    '2': 'D',
    'b3': 'D#',
    '3': 'E',
    '4': 'F',
    '#4': 'F#',
    '5': 'G',
    'b6': 'G#',
    '6': 'A',
    'b7': 'A#',
    '7': 'B',
}

scale_start = 0

Label = Union[str, Tuple[str, str], dict]

# Modal colours (1–7)
mode_colors = {
    1: "#90C978",  # Ionian - calm green
    2: "#e58c19",  # Dorian - gold
    3: "#f6dd36",  # Phrygian - bright yellow
    4: "#4ED1C5",  # Lydian - bright turquoise
    5: "#A080C6",  # Mixolydian - purple
    6: "#6AA8D8",  # Aeolian - stable blue
    7: "#D14A4A",  # Locrian - fierce red
}


def _parse_color(c, default=None):
    if c is None:
        return default
    try:
        return ImageColor.getrgb(c)
    except Exception:
        return default


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _adjust_hsl(rgb: Tuple[int, int, int], sat_mult=1.0, light_add=0.0, light_mult=1.0):
    r, g, b = [v / 255.0 for v in rgb]
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # HLS
    s = _clamp(s * float(sat_mult))
    l = _clamp(l * float(light_mult) + float(light_add))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (int(round(r2 * 255)), int(round(g2 * 255)), int(round(b2 * 255)))


def _get_float(d: dict, keys: list, default: float) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return default


def _normalize_labels(
        labels: List[Label],
        default_fill,
        default_text,
        default_outline,
        default_blend="normal"
):
    """Parses labels and captures optional per-label modifiers and placement:
       Placement:
         - sequential grid (no xy/rc)
         - rc-indexed grid {rc:(row,col)} or {row:r,col:c}  (zero-based, includes extra-left column as col 0)
         - absolute {xy:(x,y)} (image pixel center)

       Style fields supported:
         text, fill, outline, text_color, blend('normal'|'darken'),
         sat/s, light/l (add), l_mult,
         box_w/box_h/box_radius, left_shift_px,
         border_width  ← per-label outline width (px, unscaled)
    """
    norm = []
    for item in labels:

        base, extras = {}, {}
        if isinstance(item, str):
            base = {"text": item, "fill": default_fill}
        elif isinstance(item, tuple) and len(item) == 2:
            t, f = item
            base = {"text": t, "fill": _parse_color(f, default_fill)}
        elif isinstance(item, dict):
            if isinstance(item["fill"], tuple) and len(item["fill"]) == 2:
                base = {"text": item.get("text", ""), "fill": (
                _parse_color(item.get("fill")[0], default_fill), _parse_color(item.get("fill")[1], default_fill))}
            else:
                base = {"text": item.get("text", ""), "fill": _parse_color(item.get("fill"), default_fill)}
            rc = item.get("rc")
            if not rc and ("row" in item and "col" in item):
                rc = (int(item["row"]), int(item["col"]))
            xy = item.get("xy")
            if not xy and ("x" in item and "y" in item):
                xy = (item["x"], item["y"])
            extras = {
                "rc": tuple(rc) if rc is not None else None,
                "xy": tuple(xy) if xy is not None else None,
                "button": bool(item.get("button", False)),
                "left_shift_px": item.get("left_shift_px", None),
                "text_color": _parse_color(item.get("text_color"), default_text),
                "outline": _parse_color(item.get("outline"), default_outline),
                "blend": (item.get("blend") or item.get("style") or default_blend).lower(),
                "sat_mult": _get_float(item, ["sat", "s", "s_mult"], 1.0),
                "light_add": _get_float(item, ["light", "l", "l_add", "l_delta"], 0.0),
                "light_mult": _get_float(item, ["l_mult"], 1.0),
                "box_w": item.get("box_w", None),
                "box_h": item.get("box_h", None),
                "box_radius": item.get("box_radius", None),
                "font_size": item.get('font_size', 32),
                # NEW: per-label border width (px, unscaled). Accept synonyms.
                "border_width": _get_float(item, ["border_width", "outline_width", "stroke_width", "border"], None),
            }
        else:
            base = {"text": str(item), "fill": default_fill, "font_size": item.get('font_size', 32), }

        blend = extras.get("blend", default_blend)
        if blend not in ("normal", "darken"):
            blend = default_blend

        norm.append({
            "text": base.get("text", ""),
            "fill": base.get("fill", default_fill),
            "text_color": extras.get("text_color", default_text),
            "outline": extras.get("outline", default_outline),
            "blend": blend,
            "sat_mult": extras.get("sat_mult", 1.0),
            "light_add": extras.get("light_add", 0.0),
            "light_mult": extras.get("light_mult", 1.0),
            "rc": extras.get("rc", None),
            "xy": extras.get("xy", None),
            "button": extras.get("button", False),
            "left_shift_px": extras.get("left_shift_px", None),
            "box_w": extras.get("box_w", None),
            "box_h": extras.get("box_h", None),
            "box_radius": extras.get("box_radius", None),
            "font_size": extras.get("font_size", None),
            "border_width": extras.get("border_width", None),  # ← store per-label border width
        })
    return norm


def add_label_grid_onto_image(
        image_path: str,
        output_path: str,
        labels: List[Label],
        *,
        rows: int,
        cols: int,  # original columns (not counting extra-left column)
        box_size: Tuple[int, int] = (70, 55),  # 55 px height as requested
        box_radius: int = 8,
        grid_origin: Tuple[int, int] = (337, 59),  # center of original col 0, row 0
        origin_step: Tuple[int, int] = (102, 72),  # center-to-center spacing (x, y)
        font_path: Optional[str] = "Helve22.ttf",
        font_size: int,
        default_text_color: str = "#000000",
        default_outline: Optional[str] = "#000000",
        outline_width: int = 2,  # global default border width (px)
        scale: int = 4,
        extra_left_column: bool = True,
        extra_left_offset_px: int = 133,  # extra column center is 133 px left of original col 0
        box_left_offset_px: int = 51,  # global left shift for grid boxes
        plus_one_corner_rounding: bool = True,
        draw_markers: bool = False,  # markers OFF by default
) -> None:
    base = Image.open(image_path).convert("RGBA")
    W, H = base.size
    S = max(1, int(scale))

    up = base.resize((W * S, H * S), Image.LANCZOS)
    draw = ImageDraw.Draw(up)

    if 'name' in labels[0]:
        name = '%s to %s' % (labels[0]['name'][0], labels[0]['name'][1])
        try:
            font = ImageFont.truetype(font_path, 32 * S) if font_path else ImageFont.load_default()
        except OSError:
            print(f"Could not load font at {font_path}, using default font.")
        draw.text((600, 2000), labels[0]['name'][0], font=font, fill='#6AA8D8')
        draw.text((400, 2100), 'to', font=font, fill=default_text_color)
        draw.text((600, 2200), labels[0]['name'][1], font=font, fill='#e58c19')

    # Scaled defaults
    def_box_w, def_box_h = box_size[0] * S, box_size[1] * S
    origin_x, origin_y = grid_origin[0] * S, grid_origin[1] * S
    step_x, step_y = origin_step[0] * S, origin_step[1] * S
    def_rad = (box_radius + (1 if plus_one_corner_rounding else 0)) * S
    grid_left_shift = box_left_offset_px * S

    d_text = _parse_color(default_text_color, (0, 0, 0))
    d_outline = _parse_color(default_outline, (0, 0, 0))

    items = _normalize_labels(labels, default_fill=(255, 238, 153), default_text=d_text,
                              default_outline=d_outline, default_blend="normal")

    # Partition by placement
    seq_items = [lab for lab in items if lab["xy"] is None and lab["rc"] is None]
    rc_items = [lab for lab in items if lab["rc"] is not None]
    abs_items = [lab for lab in items if lab["xy"] is not None]

    # Visible grid with optional extra-left col (zero-based: col 0 is the extra-left column)
    extra_cols = 1 if extra_left_column else 0
    total_cols = cols + extra_cols
    extra_dx = extra_left_offset_px * S

    def col_center_x(vcol: int) -> int:
        if extra_left_column:
            return (origin_x - extra_dx) if vcol == 0 else (origin_x + (vcol - 1) * step_x)
        return origin_x + vcol * step_x

    # Helper: get scaled border width for a label
    def _bw(lab) -> int:
        bw_unscaled = lab.get("border_width", None)
        base_bw = outline_width if (bw_unscaled is None) else bw_unscaled
        return max(1, int(round(base_bw * S)))

    # A) Sequential row-major fill (no rc/xy)
    gi = 0
    for r in range(rows):
        for vcol in range(total_cols):
            if gi >= len(seq_items):
                break
            cx, cy = col_center_x(vcol), origin_y + r * step_y
            lab = seq_items[gi];
            gi += 1

            adj = _adjust_hsl(lab["fill"], lab.get("sat_mult", 1.0), lab.get("light_add", 0.0),
                              lab.get("light_mult", 1.0))
            bx_w = lab.get("box_w") or def_box_w
            bx_h = lab.get("box_h") or def_box_h
            rad = lab.get("box_radius") or def_rad
            x0 = cx - bx_w // 2 - grid_left_shift
            y0 = cy - bx_h // 2
            x1, y1 = x0 + bx_w, y0 + bx_h
            bw = _bw(lab)

            if lab.get("blend", "normal") == "darken":
                overlay = up.copy();
                d2 = ImageDraw.Draw(overlay)
                d2.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=adj)
                up = ImageChops.darker(up, overlay);
                draw = ImageDraw.Draw(up)
                draw.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=None, outline=lab["outline"], width=bw)
            else:
                draw.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=adj, outline=lab["outline"], width=bw)

            if lab["text"]:
                # Font
                try:
                    font = ImageFont.truetype(font_path,
                                              lab['font_size'] * S) if font_path else ImageFont.load_default()
                except OSError:
                    print(f"Could not load font at {font_path}, using default font.")
                    font = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), lab["text"], font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = x0 + (bx_w - tw) // 2 - bbox[0]
                ty = y0 + (bx_h - th) // 2 - bbox[1]
                draw.text((tx, ty), lab["text"], font=font, fill=lab["text_color"])

    # B) rc-indexed labels (row/col)
    for lab in rc_items:
        # print(lab)
        r, vcol = int(lab["rc"][0]), int(lab["rc"][1])
        if not (0 <= r < rows and 0 <= vcol < total_cols):
            continue
        cx, cy = col_center_x(vcol), origin_y + r * step_y

        if lab.get("button", False):
            # print(lab['fill'])
            if isinstance(lab["fill"], tuple) and len(lab["fill"]) == 2:
                adj1 = _adjust_hsl(lab["fill"][0], lab.get("sat_mult", 1.0), lab.get("light_add", 0.0),
                                   lab.get("light_mult", 1.0))
                adj2 = _adjust_hsl(lab["fill"][1], lab.get("sat_mult", 1.0), lab.get("light_add", 0.0),
                                   lab.get("light_mult", 1.0))
            else:
                adj1 = _adjust_hsl(lab["fill"], lab.get("sat_mult", 1.0), lab.get("light_add", 0.0),
                                   lab.get("light_mult", 1.0))
                adj2 = adj1

            bx_w = lab.get("box_w") or def_box_w
            bx_h = lab.get("box_h") or def_box_h
            rad = lab.get("box_radius") or def_rad
            lshift = grid_left_shift if lab.get("left_shift_px", None) is None else int(round(lab["left_shift_px"] * S))
            x0 = cx - bx_w // 2 - lshift
            y0 = cy - bx_h // 2
            x1, y1 = x0 + bx_w, y0 + bx_h
            bw = _bw(lab)

            if lab.get("blend", "normal") == "darken":
                overlay = up.copy();
                d2 = ImageDraw.Draw(overlay)
                d2.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=adj1)
                up = ImageChops.darker(up, overlay);
                draw = ImageDraw.Draw(up)
                draw.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=None, outline=lab["outline"], width=bw)
            else:

                w = x1 - x0
                h = y1 - y0

                if isinstance(lab["fill"], tuple) and len(lab["fill"]) == 2:
                    c1 = adj1
                    c2 = adj2

                    # force RGBA explicitly
                    box1 = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    box2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))

                    ImageDraw.Draw(box1).rounded_rectangle(
                        (0, 0, w, h), radius=rad, fill=c1
                    )
                    ImageDraw.Draw(box2).rounded_rectangle(
                        (0, 0, w, h), radius=rad, fill=c2
                    )

                    mask = Image.new("L", (w, h), 0)
                    ImageDraw.Draw(mask).polygon(
                        [(0, 0), (w, 0), (0, h)], fill=255
                    )

                    split_box = Image.composite(box1, box2, mask)

                    up.paste(split_box, (x0, y0), split_box)

                    draw.rounded_rectangle(
                        (x0, y0, x1, y1),
                        radius=rad,
                        outline=lab["outline"],
                        width=bw
                    )

                else:
                    draw.rounded_rectangle(
                        (x0, y0, x1, y1),
                        radius=rad,
                        fill=adj1,
                        outline=lab["outline"],
                        width=bw
                    )
            if lab["text"]:

                try:
                    font = ImageFont.truetype(font_path,
                                              lab['font_size'] * S) if font_path else ImageFont.load_default()
                except OSError:
                    print(f"Could not load font at {font_path}, using default font.")
                bbox = draw.textbbox((0, 0), lab["text"], font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = x0 + (bx_w - tw) // 2 - bbox[0]
                ty = y0 + (bx_h - th) // 2 - bbox[1]
                draw.text((tx, ty), lab["text"], font=font, fill=lab["text_color"])

    # C) absolute xy labels
    for lab in abs_items:
        cx, cy = int(round(lab["xy"][0] * S)), int(round(lab["xy"][1] * S))
        if lab.get("button", False):
            adj = _adjust_hsl(lab["fill"], lab.get("sat_mult", 1.0), lab.get("light_add", 0.0),
                              lab.get("light_mult", 1.0))
            bx_w = lab.get("box_w") or def_box_w
            bx_h = lab.get("box_h") or def_box_h
            rad = lab.get("box_radius") or def_rad
            lshift = int(round((lab.get("left_shift_px", 0) or 0) * S))
            x0 = cx - bx_w // 2 - lshift
            y0 = cy - bx_h // 2
            x1, y1 = x0 + bx_w, y0 + bx_h
            bw = _bw(lab)

            if lab.get("blend", "normal") == "darken":
                overlay = up.copy();
                d2 = ImageDraw.Draw(overlay)
                d2.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=adj)
                up = ImageChops.darker(up, overlay);
                draw = ImageDraw.Draw(up)
                draw.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=None, outline=lab["outline"], width=bw)
            else:
                draw.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=adj, outline=lab["outline"], width=bw)

            if lab["text"]:
                try:
                    font = ImageFont.truetype(font_path,
                                              lab['font_size'] * S) if font_path else ImageFont.load_default()
                except OSError:
                    print(f"Could not load font at {font_path}, using default font.")
                bbox = draw.textbbox((0, 0), lab["text"], font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = x0 + (bx_w - tw) // 2 - bbox[0]
                ty = y0 + (bx_h - th) // 2 - bbox[1]
                draw.text((tx, ty), lab["text"], font=font, fill=lab["text_color"])

    # Downsample (AA)
    result = up.resize((W, H), Image.LANCZOS)
    result.save(output_path)


# Caged System data

E = """
  - note: 7
    pos: [6,0]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [6,1]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [6,3]
    is_pent: True
    is_arp: False     
  - note: 3
    pos: [5,0]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [5,1]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [5,3]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [4,0]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [4,2]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [4,3]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [3,0]
    is_pent: True
    is_arp: False     
  - note: 3
    pos: [3,2]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [3,3]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [2,1]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [2,3]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [1,0]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [1,1]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [1,3]
    is_pent: True
    is_arp: False    
"""

D = """
  - note: 2
    pos: [6,3]
    is_pent: True
    is_arp: False 
  - note: 3
    pos: [6,5]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [6,6]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [5,3]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [5,5]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [4,2]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [4,3]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [4,5]
    is_pent: True
    is_arp: False     
  - note: 3
    pos: [3,2]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [3,3]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [3,5]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [2,3]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [2,5]
    is_pent: False 
    is_arp: False
  - note: 1
    pos: [2,6]
    is_pent: True
    is_arp: True     
  - note: 2
    pos: [1,3]
    is_pent: True
    is_arp: False
  - note: 3
    pos: [1,5]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [1,6]
    is_pent: False
    is_arp: False
"""

C = """
  - note: 3
    pos: [6,5]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [6,6]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [6,8]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [5,5]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [5,7]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [5,8]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [4,5]
    is_pent: True
    is_arp: False     
  - note: 3
    pos: [4,7]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [4,8]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [3,5]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [3,7]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [2,5]
    is_pent: False 
    is_arp: False
  - note: 1
    pos: [2,6]
    is_pent: True
    is_arp: True     
  - note: 2
    pos: [2,8]
    is_pent: True
    is_arp: False
  - note: 3
    pos: [1,5]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [1,6]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [1,8]
    is_pent: True
    is_arp: True
"""

A = """
  - note: 5
    pos: [6,8]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [6,10]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [5,7]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [5,8]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [5,10]
    is_pent: True
    is_arp: False     
  - note: 3
    pos: [4,7]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [4,8]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [4,10]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [3,7]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [3,9]
    is_pent: False 
    is_arp: False
  - note: 1
    pos: [3,10]
    is_pent: True
    is_arp: True     
  - note: 2
    pos: [2,8]
    is_pent: True
    is_arp: False
  - note: 3
    pos: [2,10]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [2,11]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [1,8]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [1,10]
    is_pent: True
    is_arp: False


"""

G = """
  - note: 6
    pos: [6,10]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [6,12]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [6,13]
    is_pent: True 
    is_arp: True
  - note: 2
    pos: [5,10]
    is_pent: True
    is_arp: False     
  - note: 3
    pos: [5,12]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [5,13]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [4,10]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [4,12]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [3,9]
    is_pent: False 
    is_arp: False
  - note: 1
    pos: [3,10]
    is_pent: True
    is_arp: True     
  - note: 2
    pos: [3,12]
    is_pent: True
    is_arp: False
  - note: 3
    pos: [2,10]
    is_pent: True
    is_arp: True
  - note: 4
    pos: [2,11]
    is_pent: False
    is_arp: False
  - note: 5
    pos: [2,13]
    is_pent: True
    is_arp: True
  - note: 6
    pos: [1,10]
    is_pent: True
    is_arp: False
  - note: 7
    pos: [1,12]
    is_pent: False
    is_arp: False
  - note: 1
    pos: [1,13]
    is_pent: True 
    is_arp: True

"""

intervals_6 = """6472
6517
6726
5436
5546
5756
4466
4676
4718
3428
3638
3748""".splitlines()

intervals_5 = """6342
6552
6762
5472
5517
5726
4436
4546
4756
3466
3676
3718
2428
2638
2748""".splitlines()

intervals_4 = """6318
6522
6732
6842
5342
5552
5762
4472
4517
4726
3436
3546
3756
2356
2566
2776
2818
1528
1738
1848""".splitlines()

intervals_3 = """5318
5522
5732
5842
4342
4552
4762
3472
3517
3726
2536
2646
2856
1356
1566
1776
1818
""".splitlines()

intervals_2 = """4318
4522
4732
4842
3342
3552
3762
2572
2617
2826
1536
1646
1856
""".splitlines()

intervals_1 = """3318
3522
3732
3842
2442
2652
2762
1572
1617
1826
""".splitlines()

triad_1 = """4,1,p5,1
5,3,Δ3,1
6,4,R,1
4,6,R,2
5,6,p5,2
6,8,Δ3,2
4,10,Δ3,3
5,11,R,3
6,11,p5,3
""".splitlines()

triad_2 = """3,1,p5,1
4,3,Δ3,1
5,4,R,1
3,6,R,2
4,6,p5,2
5,8,Δ3,2
3,10,Δ3,3
4,11,R,3
5,11,p5,3
""".splitlines()

triad_3 = """2,2,p5,1
3,3,Δ3,1
4,4,R,1
2,7,R,2
3,6,p5,2
4,8,Δ3,2
2,11,Δ3,3
3,11,R,3
4,11,p5,3
""".splitlines()

triad_4 = """1,2,p5,1
2,4,Δ3,1
3,4,R,1
1,7,R,2
2,7,p5,2
3,8,Δ3,2
1,11,Δ3,3
2,12,R,3
3,11,p5,3
""".splitlines()

triad_5 = """1,5,R,1
2,2,Δ3,1
4,2,p5,1
1,12,p5,2
2,10,R,2
4,11,Δ3,2
1,9,Δ3,3
2,5,p5,3
4,7,R,3
""".splitlines()

triad_6 = """2,5,p5,1
3,2,R,1
5,4,Δ3,1
2,10,R,2
3,6,Δ3,2
5,7,p5,2
2,14,Δ3,3
3,9,p5,3
5,12,R,3
""".splitlines()

triad_7 = """3,6,Δ3,1
4,2,p5,1
6,5,R,1
3,9,p5,2
4,7,R,2
6,9,Δ3,2
3,14,R,3
4,11,Δ3,3
6,12,p5,3
""".splitlines()

triad_8 = """1,5,R,1
3,6,Δ3,1
4,2,p5,1
1,9,Δ3,2
3,9,p5,2
4,7,R,2
1,12,p5,3
3,14,R,3
4,11,Δ3,3
""".splitlines()

triad_9 = """2,2,Δ3,1
4,2,p5,1
5,0,R,1
2,5,p5,2
4,7,R,2
5,4,Δ3,2
2,10,R,3
4,11,Δ3,3
5,7,p5,3
""".splitlines()

triad_10 = """3,2,R,1
5,4,Δ3,1
6,0,p5,1
3,6,Δ3,2
5,7,p5,2
6,5,R,2
3,9,p5,3
5,12,R,3
6,9,Δ3,3
""".splitlines()

triad_11 = """1,0,p5,1
3,2,R,1
5,4,Δ3,1
1,5,R,2
3,6,Δ3,2
5,7,p5,2
1,9,Δ3,3
3,9,p5,3
5,12,R,3
""".splitlines()
triad_12 = """2,2,Δ3,1
4,2,p5,1
6,5,R,1
2,5,p5,2
4,7,R,2
6,9,Δ3,2
2,10,R,3
4,11,Δ3,3
6,12,p5,3
""".splitlines()

# string (6 is low E),fret,name,colour
seventh_1 = """6,1,R,1
5,3,p5,1
4,2,Δ7,1
3,2,Δ3,1
6,6,Δ3,2
5,8,Δ7,2
4,4,R,2
3,6,p5,2
6,11,p5,3
5,11,R,3
4,10,Δ3,3
3,12,Δ7,3
6,15,Δ7,4
5,15,Δ3,4
4,13,p5,4
3,13,R,4
""".splitlines()

seventh_2 = """5,1,R,1
4,3,p5,1
3,2,Δ7,1
2,3,Δ3,1
5,6,Δ3,2
4,8,Δ7,2
3,4,R,2
2,7,p5,2
5,11,p5,3
4,11,R,3
3,10,Δ3,3
2,13,Δ7,3
5,15,Δ7,4
4,15,Δ3,4
3,13,p5,4
2,14,R,4
""".splitlines()

seventh_3 = """4,1,R,1
3,3,p5,1
2,3,Δ7,1
1,3,Δ3,1
4,6,Δ3,2
3,8,Δ7,2
2,5,R,2
1,7,p5,2
4,11,p5,3
3,11,R,3
2,11,Δ3,3
1,13,Δ7,3
4,15,Δ7,4
3,15,Δ3,4
2,14,p5,4
1,14,R,4
""".splitlines()

seventh_4 = """6,1,R,1
4,2,Δ7,1
3,2,Δ3,1
2,1,p5,1
6,5,Δ3,2
4,3,R,2
3,5,p5,2
2,5,Δ7,2
6,8,p5,3
4,7,Δ3,3
3,9,Δ7,3
2,6,R,3
6,12,Δ7,4
4,10,p5,4
3,10,R,4
2,10,Δ3,4
""".splitlines()

seventh_5 = """5,1,R,1
3,2,Δ7,1
2,3,Δ3,1
1,1,p5,1
5,5,Δ3,2
3,3,R,2
2,6,p5,2
1,5,Δ7,2
5,8,p5,3
3,7,Δ3,3
2,10,Δ7,3
1,6,R,3
5,12,Δ7,4
3,10,p5,4
2,11,R,4
1,10,Δ3,4
""".splitlines()

seventh_6 = """6,1,R,1
5,3,p5,1
3,2,Δ3,1
2,5,Δ7,1
6,5,Δ3,2
5,7,Δ7,2
3,5,p5,2
2,6,R,2
6,8,p5,3
5,8,R,3
3,9,Δ7,3
2,10,Δ3,3
6,12,Δ7,4
5,12,Δ3,4
3,10,R,4
2,13,p5,4
""".splitlines()

seventh_7 = """5,1,R,1
4,3,p5,1
2,3,Δ3,1
1,5,Δ7,1
5,5,Δ3,2
4,7,Δ7,2
2,6,p5,2
1,6,R,2
5,8,p5,3
4,8,R,3
2,10,Δ7,3
1,10,Δ3,3
5,12,Δ7,4
4,12,Δ3,4
2,11,R,4
1,13,p5,4
""".splitlines()


def make(i, mod):
    print(i)
    mapping = {
        'b3': (3, -1),
        'b7': (7, -1),
    }

    note_colors = {
        0: '#666666',
        1: "#90C978",  # Ionian - calm green
        2: "#e58c19",  # Dorian - gold
        3: "#f6dd36",  # Phrygian - bright yellow
        4: "#4ED1C5",  # Lydian - bright turquoise
        5: "#A080C6",  # Mixolydian - purple
        6: "#6AA8D8",  # Aeolian - stable blue
        7: "#D14A4A",  # Locrian - fierce red
        8: "#333333",  # dark grey
        9: "#FFFFFF",  # white
    }

    labels = []
    for row in range(1, 7):
        for fret in range(0, 16):
            for n in i:
                if ',' in n:

                    r, f, note, col = n.split(',')
                    r = int(r)
                    f = int(f)
                    col = int(col)

                    strnote = str(note)
                else:
                    r = int(n[0])
                    f = int(n[1])
                    note = int(n[2])
                    strnote = str(note)
                    col = int(n[3])

                if row == r and fret == f:
                    Y = row - 1
                    X = fret
                    border = 3
                    light = 0
                    sat = 1
                    color = note_colors[int(col)]

                    for k, map in mapping.items():
                        for m in mod:
                            if m == k and note == map[0]:
                                strnote = k
                                X = X + map[1]
                    pos = (Y, X)
                    blend = None
                    outline = '#000000'
                    text_color = '#000000'

                    label = {"rc": pos, "text": strnote, "fill": color, 'button': True, 'sat': sat, 'light': light,
                             "border_width": border, 'blend': blend, 'outline': outline, 'text_color': text_color}
                    labels.append(label)

    return labels


def read_row(n):
    r, f, note, col = n.split(',')
    r = int(r)
    f = int(f)
    col = int(col)
    return {'r': r, 'f': f, 'note': note, 'col': col}


def ORM(triad):
    t1 = [read_row(t) for t in triad[0:3]]
    t2 = [read_row(t) for t in triad[3:6]]
    t3 = [read_row(t) for t in triad[6:9]]

    print('1:', t1)
    print('2:', t2)
    print('3:', t3)
    # if root triad
    if 'R' in t1[2]['note']:
        t_root = t1
    elif 'R' in t2[2]['note']:
        t_root = t2
    elif 'R' in t3[2]['note']:
        t_root = t3

    if '3' in t1[2]['note']:
        t_first = t1
    elif '3' in t2[2]['note']:
        t_first = t2
    elif '3' in t3[2]['note']:
        t_first = t3

    if '5' in t1[2]['note']:
        t_second = t1
    elif '5' in t2[2]['note']:
        t_second = t2
    elif '5' in t3[2]['note']:
        t_second = t3

    print('ROOT', t_root)
    print('FIRST', t_first)
    print('SECOND', t_second)

    root_offset = 3 - t_root[2]['f']

    for t in t_root:
        t['f'] = t['f'] + root_offset
        t['col'] = '1'

    first_offset = 7 - t_first[2]['f']

    for t in t_first:
        t['f'] = t['f'] + first_offset
        t['col'] = '2'

    second_offset = 11 - t_second[2]['f']

    for t in t_second:
        t['f'] = t['f'] + second_offset
        print(t['f'])
        t['col'] = '3'

    labels = []

    for t in t_root + t_first + t_second:
        labels.append(f"{t['r']},{t['f']},{t['note']},{t['col']}")

    return labels


def make_noteboard():
    labels = []
    notes_mapping = {
        '1': 'C',
        'b2': 'C#',
        '2': 'D',
        'b3': 'D#',
        '3': 'E',
        '4': 'F',
        '#4': 'F#',
        '5': 'G',
        'b6': 'G#',
        '6': 'A',
        'b7': 'A#',
        '7': 'B',
    }
    string_step = [5, 4, 5, 5, 5, 5]
    accumulated_step = [24, 19, 15, 10, 5, 0]

    # C, C#, D, D#, E, F, F#, G, G#, A, A#, B

    tone = ['3', '4', '#4', '5', 'b6', '6', 'b7', '7', '1', 'b2', '2', 'b3']

    note_colors = {
        0: '#666666',
        1: "#90C978",  # Ionian - calm green
        2: "#e58c19",  # Dorian - gold
        3: "#f6dd36",  # Phrygian - bright yellow
        4: "#4ED1C5",  # Lydian - bright turquoise
        5: "#A080C6",  # Mixolydian - purple
        6: "#6AA8D8",  # Aeolian - stable blue
        7: "#D14A4A",  # Locrian - fierce red
    }

    for row in range(1, 7):
        for fret in range(0, 16):

            Y = 6 - row
            X = fret

            N = fret + accumulated_step[Y]

            note = tone[N % 12]

            if 'b' in notes_mapping[note] or '#' in notes_mapping[note]:
                border = 1
                light = 0
                sat = 1
                color = '#FFFFFF'

                pos = (Y, X)
                blend = 'darken'
                outline = '#999999'
                text_color = '#666666'
            else:
                border = 3
                light = 0
                sat = 1
                color = note_colors[int(note)]

                pos = (Y, X)
                blend = None
                outline = '#000000'
                text_color = '#000000'

            label = {"rc": pos, "text": notes_mapping[note], "fill": color, 'button': True, 'sat': sat, 'light': light,
                     "border_width": border, 'blend': blend, 'outline': outline, 'text_color': text_color}
            labels.append(label)

    return labels


def make_caged(y, mode=1, start=1):
    data = yaml.safe_load(y)
    labels = []

    for v in data:
        color = mode_colors[mode]
        pos = (v['pos'][0] - 1, v['pos'][1])
        text = str(v['note'])
        is_pent = v['is_pent']
        is_arp = v['is_arp']
        if is_pent:
            sat = 1.1
            light = 0
            border = 4
        else:
            sat = 0.2
            border = 1
            light = -0.2

        if is_arp:
            sat = 3
        else:
            light = -0.2

        label = {"rc": pos, "text": text, "fill": color, 'button': True, 'sat': sat, 'light': light,
                 "border_width": border}
        labels.append(label)


    return labels


def make_mode(mode, root=True, degree=False):
    row = 6
    fret = scale_start + scale[mode - 1]
    count = 0
    offset = 0
    if not root:
        offset = scale[mode - 1]

    labels = []
    for string in [6, 5, 4, 3, 2, 1]:
        for nps in [1, 2, 3]:
            i = (nps - 1) + (row - 1) * cols
            color = mode_colors[mode]

            if degree:
                text = tone[offset % 12]
            else:
                text = str(count % 7 + 1)
            label = {"rc": (row - 1, fret), "text": text, "fill": color, 'button': True}
            labels.append(label)

            fret = fret + scale_step[(count + (mode - 1)) % 7]
            offset += scale_step[(count + (mode - 1)) % 7]
            count += 1
            if nps == 3:
                row = row - 1
                fret = fret - string_step[row - 1]

    return labels

def chord_notes(note, type, extensions, name):
    tone = ['1', 'b9', '9', 'b3', '3', '11', 'b5','5', 'b13', '13', 'b7', '7', ]
    notes_mapping = {
        '1': 'C',
        'b2': 'C#',
        '2': 'D',
        'b3': 'D#',
        '3': 'E',
        '4': 'F',
        '#4': 'F#',
        '5': 'G',
        'b6': 'G#',
        '6': 'A',
        'b7': 'A#',
        '7': 'B',
    }
    chromatic_mapping = {
        'Ab': 12,
        'A': 1,
        'A#': 2,
        'Bb': 2,
        'B': 3,
        'B#': 4,
        'Cb': 3,
        'C': 4,
        'C#': 5,
        'Db': 5,
        'D': 6,
        'D#': 7,
        'Eb': 7,
        'E': 8,
        'E#': 9,
        'F': 9,
        'F#': 10,
        'Gb': 10,
        'G': 11,
        'G#': 12,
    }

    types = {
        'm': [1, 4, 8],
        'M': [1, 5, 8],
        'm7': [1, 4, 8, 11],
        'M7': [1, 5, 8, 11],
        '7': [1, 5, 8, 11],
        'maj7': [1, 5, 8, 12],
        'm7b5': [1, 4, 7, 11],
    }

    extensions_mapping = {
        '#5': 9,
        'b5': 7,
        '9': 3,
        '11': 6,
        'b13': 9,
        '13': 10,
    }

    note_offset = chromatic_mapping[note]
    notes = types[type]
    chord = []
    for n in notes:
        chord.append( (n + note_offset) % 12)

    for e in extensions:
        chord.append( (extensions_mapping[e] + note_offset) % 12)

    # for c in chord:
    #     print(tone[(c-note_offset)%12-1])
    # print(chord)
    # import sys
    # sys.exit()
    return note_offset, chord, name

def print_chord(chord):
    for c in chord[1]:
        print(tone[(c-chord[0])%12-1], chord[0], c)

def mark_chord_change(current_chord, next_chord):
    # print('>>>')
    # print_chord(current_chord)
    # print('---')
    # print_chord(next_chord)
    # print('<<<')

    current = current_chord[1]
    current_offset = current_chord[0]
    current_name = current_chord[2]
    next = next_chord[1]
    next_offset = next_chord[0]
    next_name = next_chord[2]


    labels = []
    notes_mapping = {
        '1': 'C',
        'b9': 'C#',
        '9': 'D',
        'b3': 'D#',
        '3': 'E',
        '11': 'F',
        'b5': 'F#',
        '5': 'G',
        'b13': 'G#',
        '13': 'A',
        'b7': 'A#',
        '7': 'B',
    }
    string_step = [5, 4, 5, 5, 5, 5]
    accumulated_step = [24, 19, 15, 10, 5, 0]

    # C, C#, D, D#, E, F, F#, G, G#, A, A#, B

    tone = ['1', 'b9', '9', 'b3', '3', '11', 'b5', '5', 'b13', '13', 'b7', '7', ]

    note_colors = {
        0: '#666666',
        1: "#90C978",  # Ionian - calm green
        2: "#e58c19",  # Dorian - gold
        3: "#f6dd36",  # Phrygian - bright yellow
        4: "#4ED1C5",  # Lydian - bright turquoise
        5: "#A080C6",  # Mixolydian - purple
        6: "#6AA8D8",  # Aeolian - stable blue
        7: "#D14A4A",  # Locrian - fierce red
    }
    chrom_note_colors = {
        0: '#666666',
        1: "#90C978",  # C
        2: "#90C978",  # C#
        3: "#e58c19",  # D
        4: "#e58c19",  # D#
        5: "#f6dd36",  # E
        6: "#4ED1C5",  # F
        7: "#4ED1C5",  # F#
        8: "#A080C6",  # G
        9: "#A080C6",  # G#
        10: "#6AA8D8",  # A
        11: "#6AA8D8",  # A#
        12: "#D14A4A",  # B
    }
    for row in range(1, 7):
        for fret in range(0, 16):

            Y = 6 - row
            X = fret

            N = fret + accumulated_step[Y]

            chrom_note = ((N+9) % 12)
            note = tone[ (19 + chrom_note) % 12 ]

            font_size = 32

            if chrom_note in current and chrom_note in next:
                border = 3
                light = 0
                sat = 1
                color = (note_colors[6],note_colors[2])
                #color = note_colors[3]

                pos = (Y, X)
                blend = None
                outline = '#000000'
                text_color = '#000000'
                text = '%s/%s'%(tone[ (11 + chrom_note-current_offset) % 12 ], tone[ (11 + chrom_note-next_offset) % 12 ])
                font_size = 24

            elif chrom_note in current:
                border = 3
                light = 0
                sat = 1
                color = note_colors[6]

                pos = (Y, X)
                blend = None
                outline = '#000000'
                text_color = '#000000'
                text = tone[ (11 + chrom_note-current_offset) % 12 ]
            elif chrom_note in next:
                border = 3
                light = 0
                sat = 1
                color = note_colors[2]

                pos = (Y, X)
                blend = None
                outline = '#000000'
                text_color = '#000000'
                text = tone[ (11 + chrom_note-next_offset) % 12 ]

            else:
                border = 1
                light = 0
                sat = 1
                color = '#FFFFFF'

                pos = (Y, X)
                blend = 'darken'
                outline = '#999999'
                text_color = '#666666'
                text = notes_mapping[note]

            # if 'b' in notes_mapping[note] or '#' in notes_mapping[note]:
            #     border = 1
            #     light = 0
            #     sat = 1
            #     color = '#FFFFFF'
            #
            #     pos = (Y, X)
            #     blend = 'darken'
            #     outline = '#999999'
            #     text_color = '#666666'
            # else:
            #     border = 3
            #     light = 0
            #     sat = 1
            #     color = note_colors[int(note)]
            #
            #     pos = (Y, X)
            #     blend = None
            #     outline = '#000000'
            #     text_color = '#000000'

            label = {"rc": pos, "text": notes_mapping[note], "fill": color, 'button': True, 'sat': sat, 'light': light,
                     "border_width": border, 'blend': blend, 'outline': outline, 'text_color': text_color}
            label = {"rc": pos, "text": text, "fill": color, 'button': True, 'sat': sat, 'light': light,
                     "border_width": border, 'blend': blend, 'outline': outline, 'text_color': text_color, 'font_size': font_size, 'name': (current_name,next_name)}
            labels.append(label)

    return labels




def build_parser():
    parser = argparse.ArgumentParser(
        description="Example argparse template with flags and positional args"
    )

    parser.add_argument(
        "-m", "--make",
        metavar="ARG",
        help="Which diagrams to build"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Optional filename override"
    )


    return parser


if __name__ == "__main__":

    parser = build_parser()
    args = parser.parse_args()

    MAKE = args.make
    verbose = args.verbose
    name = args.name

    if MAKE == 'caged':
        mod = []
        label_set = []
        # a = make_mode(1, root=False, degree=True)
        # b = make_mode(4, root=True, degree=True)
        # labels = a + b
        # label_set.append(labels)

        e = make_caged(E, mode=1, start=1)
        c = make_caged(C, mode=3, start=1)
        g = make_caged(G, mode=5, start=1)
        labels = e + c + g
        label_set.append(labels)

        d = make_caged(D, mode=2, start=1)
        a = make_caged(A, mode=4, start=1)
        labels = d + a
        label_set.append(labels)

        identifier = MAKE



    if MAKE == 'noteboard':
        mod = []
        noteboard = make_noteboard()
        labels = noteboard
        identifier = MAKE

        # extas on noteboard?
        labels += [
            {"rc": (0, 0), "button": True, "text": "1", "fill": mode_colors[1], "blend": "darken"},  # extra-left column
            {"rc": (2, 7), "button": True, "text": "3", "fill": mode_colors[3], "blend": "darken",
             "sat": 0.85, "light": 0.06},  # HSL tweaks
            {"rc": (5, 15), "button": True, "text": "6", "fill": mode_colors[6], "sat": 0.6},  # desaturated
        ]
        label_set = [labels]

    if MAKE == 'intervals':
        mod = ['b7']
        label_set = [
            make(intervals_1, mod),
            make(intervals_2, mod),
            make(intervals_3, mod),
            make(intervals_4, mod),
            make(intervals_5, mod),
            make(intervals_6, mod),
        ]

    if MAKE == 'major_notes':
        mod = []
        triads = [
            triad_5,
            triad_6,
            triad_7,
            triad_8,
            triad_9,
            triad_10,
        ]

        new_labels = []
        collector = {}
        for i in triads:
            for n in i:
                if ',' in n:
                    r, f, note, col = n.split(',')
                    r = int(r)
                    f = int(f)
                    col = int(col)

                    strnote = str(note)
                else:
                    r = int(n[0])
                    f = int(n[1])
                    note = int(n[2])
                    strnote = str(note)
                    col = int(n[3])

                if not (r, f) in collector:
                    if '5' in strnote:
                        col = 3
                    if '3' in strnote:
                        col = 2
                    if 'R' in strnote:
                        col = 1

                    collector[(r, f)] = (strnote, col)

        for k, v in collector.items():
            r = str(k[0])
            f = str(k[1])
            strnote = str(v[0])
            col = str(v[1])
            new_labels.append(f'{r},{f},{strnote},{col}')

        new_labels.append('1,0,p5,3')
        new_labels.append('4,14,p5,3')

        label_set = [make(new_labels, '')]

    # triads and some skip traids
    if MAKE == 'triads':
        mod = []
        label_set = [
            make(triad_1, ''),
            make(triad_2, ''),
            make(triad_3, ''),
            make(triad_4, ''),
            make(triad_5, ''),
            make(triad_6, ''),
            make(triad_7, ''),
            make(triad_8, ''),
            make(triad_9, ''),
            make(triad_10, ''),

        ]

    # skip triads reorganised to have consistent orders
    if MAKE == 'ORM':
        mod = []
        label_set = [
            make(ORM(triad_5), ''),
            make(ORM(triad_6), ''),
            make(ORM(triad_7), ''),
            make(ORM(triad_8), ''),
            make(ORM(triad_9), ''),
            make(ORM(triad_10), ''),
            make(ORM(triad_11), ''),
            make(ORM(triad_12), ''),

        ]

    # four notes 7ths
    if MAKE == '7s':
        mod = []
        label_set = [
            make(seventh_1, ''),
            make(seventh_2, ''),
            make(seventh_3, ''),
            make(seventh_4, ''),
            make(seventh_5, ''),
            make(seventh_6, ''),
            make(seventh_7, ''),
        ]








    # Take ordered chords and make changes showing common and unshared notes
    if MAKE == 'changes':
        mod = []

        ## The basic form of creating a change is to build the label_set list as follows
        ## But I've created a changes list which then builds each transition as you'll see further down.
        # label_set = [
        #     make_changes('Cm','Fm'),
        # ]


        ## First test without extensions
        ## Format ('note', 'type', [extensions], 'display name')
        ## Types are up on line 1063 at the moment and you can pick from below. Feel free to add more you just need
        ## the chromatic note set.
        # 'm': [1, 4, 8],
        # 'M': [1, 5, 8],
        # 'm7': [1, 4, 8, 11],
        # 'M7': [1, 5, 8, 11],
        # '7': [1, 5, 8, 11],
        # 'maj7': [1, 5, 8, 12],
        # 'm7b5': [1, 4, 7, 11],

        # Cm7 = chord_notes('C', 'm7', [], 'Cm7')
        # Fm7 = chord_notes('F', 'm7', [], 'Fm7')
        # Dm7b5 = chord_notes('D', 'm7b5', [], 'Dm7b5')
        # G7 = chord_notes('G', '7', [], 'G7')
        #
        # Ebm7 = chord_notes('Eb', 'm7', [], 'Ebm7')
        # Ab7 = chord_notes('Ab', '7', [], 'Ab7')
        # Dbmaj7 = chord_notes('Db', 'maj7', [], 'Dbmaj7')
        #
        # changes = [ Cm7, Fm7, Dm7b5, G7, Cm7, Ebm7, Ab7, Dbmaj7 ]


        # This builds the note set based on chromatic numbering
        Cm7 = chord_notes('C', 'm7', ['9'], 'Cm9')
        Fm7 = chord_notes('F', 'm7', ['9'], 'Fm9')
        Dm7b5 = chord_notes('D', 'm7b5', [], 'Dm7b5')
        G7 = chord_notes('G', '7', ['b13'], 'G7(b13)')

        Ebm7 = chord_notes('Eb', 'm7', ['9'], 'Ebm9')
        Ab7 = chord_notes('Ab', '7', ['13'], 'Ab13')
        Dbmaj7 = chord_notes('Db', 'maj7', [], 'Dbmaj7')

        # The system works on changes
        changes = [ Cm7, Fm7, Dm7b5, G7, Cm7, Ebm7, Ab7, Dbmaj7, Dm7b5 ]

        label_set = []

        for n, c in enumerate(changes):
            if n < len(changes)-1:
                label_set.append( mark_chord_change(changes[n], changes[n+1]) )














    for n, labels in enumerate(label_set):
        output = f"fretboard_with_{MAKE}_{n + 1}_{''.join(mod)}.png"

        add_label_grid_onto_image(
            image_path="fretboard_withmarkers.png",
            output_path=output,
            labels=labels,
            rows=rows,
            cols=cols,
            box_size=(70, 55),
            box_radius=8,
            grid_origin=(337, 59),
            origin_step=(102, 72),
            font_path="Arial Bold.ttf",
            font_size=32,
            default_outline="#000000",
            outline_width=2,
            scale=4,
            extra_left_column=True,
            extra_left_offset_px=133,
            box_left_offset_px=51,
            plus_one_corner_rounding=True,
            draw_markers=False,  # ← marker dots disabled
        )
        print(f"Saved: {output}")

