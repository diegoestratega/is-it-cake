from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import random
import base64
import io
import numpy as np
from PIL import Image
from urllib.parse import quote
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

ALL_CAKE_THEMES = [
    "red velvet cake slice close up",
    "chocolate fudge cake close up",
    "strawberry shortcake macro",
    "white fondant wedding cake close up",
    "rainbow layer cake slice macro",
    "black forest cherry cake close up",
    "lemon drizzle cake macro",
    "carrot cake cream cheese frosting close up",
    "tres leches cake close up",
    "opera cake coffee glaze macro",
    "mango mousse mirror glaze cake close up",
    "pink ombre buttercream cake macro",
    "geode crystal cake close up",
    "naked rustic cake close up",
    "tiramisu cake close up",
    "burnt basque cheesecake macro",
    "matcha green tea cake roll close up",
    "blueberry lavender cake macro",
    "gold leaf metallic cake close up",
    "unicorn cake pastel swirls macro",
    "chocolate drip cake close up",
    "Korean bento cake close up",
    "swiss roll jelly cake close up",
    "marble bundt cake glaze macro",
    "pumpkin spice cake close up",
    "pistachio rose water cake macro",
    "blue velvet cake slice close up",
    "coconut lime cake macro",
    "dark chocolate ganache cake close up",
    "peach upside down cake macro",
    "vanilla layer cake frosting close up",
    "hazelnut praline cake close up",
    "cotton candy birthday cake macro",
    "espresso coffee layer cake close up",
    "pineapple upside down cake macro",
    "pecan caramel cake close up",
    "cherry blossom sakura cake macro",
    "galaxy mirror glaze cake close up",
    "strawberry drip birthday cake macro",
    "caramel apple cake close up",
    "funfetti birthday cake sprinkles macro",
    "mocha almond fudge cake close up",
    "banana foster cake macro",
    "orange creamsicle cake close up",
    "honey lavender berry cake macro",
    "lotus biscoff caramel cake close up",
    "red rose fondant cake macro",
    "chocolate lava cake close up",
    "cinnamon roll cake macro",
    "s'mores cake close up",
    "strawberry mousse cake macro",
    "raspberry chocolate cake close up",
    "lemon curd tart macro",
    "caramel macchiato cake close up",
    "white chocolate raspberry cake macro",
    "blackberry lavender cheesecake close up",
    "passion fruit mousse cake macro",
    "pistachio cardamom cake close up",
    "fig honey almond cake macro",
    "peanut butter chocolate cake close up",
    "salted caramel pretzel cake macro",
    "churro cake cinnamon sugar close up",
    "mochi ice cream cake macro",
    "earl grey lavender cake close up",
    "brown butter toffee cake macro",
    "key lime pie close up",
    "tiramisu cheesecake macro",
    "strawberry lemon poppy seed cake close up",
    "chocolate peppermint cake macro",
    "gingerbread spice cake close up",
    "apple cinnamon crumble cake macro",
    "cherry almond cake close up",
    "rose lychee cake macro",
    "dark chocolate espresso cake close up",
    "passionfruit mango tart macro",
    "dulce de leche layer cake close up",
    "brown sugar cinnamon cake macro",
    "raspberry rose cake close up",
    "tahini chocolate cake macro",
    "cardamom orange cake close up",
    "praline peach cake macro",
    "blood orange olive oil cake close up",
    "toasted coconut lime cheesecake macro",
    "pineapple coconut cream cake close up",
    "black sesame cake macro",
    "yuzu citrus cake close up",
    "violet blueberry cake macro",
    "smoked butterscotch cake close up",
    "cherry jubilee layer cake macro",
    "chocolate hazelnut roulade close up",
    "vanilla bean panna cotta cake macro",
    "red currant almond cake close up",
    "caramel popcorn cake macro",
    "lemon blueberry layer cake close up",
    "strawberry basil cake macro",
    "chocolate orange zest cake close up",
    "rose water pistachio cake macro",
    "maple walnut layer cake close up",
    "mint chocolate chip cake macro",
]

REAL_BACKUPS = [
    "https://images.pexels.com/photos/1055272/pexels-photo-1055272.jpeg?auto=compress&cs=tinysrgb&w=800",
    "https://images.pexels.com/photos/1721932/pexels-photo-1721932.jpeg?auto=compress&cs=tinysrgb&w=800",
    "https://images.pexels.com/photos/291528/pexels-photo-291528.jpeg?auto=compress&cs=tinysrgb&w=800",
]


def pick_theme():
    return random.choice(ALL_CAKE_THEMES)


def download_image(url):
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")
    img = Image.open(io.BytesIO(response.content)).convert("RGB")
    return img


def find_cake_probability(arr):
    h, w = arr.shape[:2]
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)

    brightness = (r + g + b) / 3.0

    not_dark       = brightness > 60
    not_too_bright = brightness < 245
    not_blue       = b < (r * 1.15 + 25)
    not_green      = g < (r * 1.25 + 15)
    warm           = r >= (b - 15)
    not_purple     = ~((b > 120) & (b > r * 1.3) & (g < r))
    not_gray       = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)) > 12

    cake_mask = not_dark & not_too_bright & not_blue & not_green & warm & not_purple & not_gray

    Y, X = np.ogrid[:h, :w]
    dx = (X - w / 2.0) / (w / 2.0)
    dy = (Y - h / 2.0) / (h / 2.0)
    center_weight = np.exp(-(dx**2 + dy**2) * 1.6)

    return cake_mask.astype(np.float32) * center_weight


def pick_cake_center(arr, region_r):
    h, w = arr.shape[:2]
    margin = region_r + 30
    prob = find_cake_probability(arr)
    prob[:margin, :]  = 0
    prob[-margin:, :] = 0
    prob[:, :margin]  = 0
    prob[:, -margin:] = 0
    total = prob.sum()

    if total < 1e-6:
        print("    ⚠️  Cake region not found — using center fallback")
        return w // 2, h // 2

    flat = prob.flatten()
    flat /= flat.sum()
    top_n = 5000
    top_idx = np.argpartition(flat, -top_n)[-top_n:]
    top_prob = flat[top_idx]
    top_prob /= top_prob.sum()
    chosen = np.random.choice(top_idx, p=top_prob)
    cy_i, cx_i = np.unravel_index(chosen, (h, w))
    return int(cx_i), int(cy_i)


def apply_subtle_change(img: Image.Image):
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    region_r = max(20, min(w, h) // 12)
    cx, cy   = pick_cake_center(arr, region_r)

    Y, X    = np.ogrid[:h, :w]
    dist_sq = (X - cx).astype(np.float32)**2 + (Y - cy).astype(np.float32)**2
    sigma   = region_r / 2.0
    mask    = np.exp(-dist_sq / (2 * sigma ** 2))
    mask3   = mask[:, :, np.newaxis]

    operation = random.choice([
        'hue_shift', 'desaturate', 'darken',
        'warm_tint', 'cool_tint', 'brighten',
    ])

    print(f"    🎨 Change: {operation} at ({cx},{cy}) radius={region_r}px [{w}x{h}]")

    modified = arr.copy()

    if operation == 'hue_shift':
        shift_deg = random.choice([32, -32, 42, -42])
        theta     = np.radians(shift_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        sq3 = np.sqrt(3)
        rot = np.array([
            [cos_t+(1-cos_t)/3,       (1-cos_t)/3-sin_t/sq3, (1-cos_t)/3+sin_t/sq3],
            [(1-cos_t)/3+sin_t/sq3,   cos_t+(1-cos_t)/3,     (1-cos_t)/3-sin_t/sq3],
            [(1-cos_t)/3-sin_t/sq3,   (1-cos_t)/3+sin_t/sq3, cos_t+(1-cos_t)/3    ]
        ], dtype=np.float32)
        norm     = arr / 255.0
        rotated  = np.clip(norm @ rot.T, 0, 1) * 255.0
        modified = arr * (1 - mask3 * 0.78) + rotated * (mask3 * 0.78)

    elif operation == 'desaturate':
        gray     = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
        gray3    = np.stack([gray, gray, gray], axis=-1)
        modified = arr * (1 - mask3 * 0.75) + gray3 * (mask3 * 0.75)

    elif operation == 'darken':
        factor   = random.uniform(0.42, 0.55)
        modified = arr * (1 - mask3 * 0.88) + (arr * factor) * (mask3 * 0.88)

    elif operation == 'brighten':
        factor   = random.uniform(1.55, 1.80)
        modified = arr * (1 - mask3 * 0.82) + np.clip(arr * factor, 0, 255) * (mask3 * 0.82)

    elif operation == 'warm_tint':
        tint     = np.full_like(arr, [255, 160, 30], dtype=np.float32)
        modified = arr * (1 - mask3 * 0.48) + tint * (mask3 * 0.48)

    elif operation == 'cool_tint':
        tint     = np.full_like(arr, [50, 110, 255], dtype=np.float32)
        modified = arr * (1 - mask3 * 0.48) + tint * (mask3 * 0.48)

    result = np.clip(modified, 0, 255).astype(np.uint8)

    return (
        Image.fromarray(result),
        round(cx / w, 4),
        round(cy / h, 4),
        round(region_r / min(w, h), 4)
    )


def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def get_real_cake_url(theme):
    print(f"[1] Fetching real cake from Pexels: '{theme}'")
    try:
        url = (
            f"https://api.pexels.com/v1/search"
            f"?query={quote(theme)}"
            f"&per_page=15"
            f"&page={random.randint(1,2)}"
            f"&size=large"
        )
        r = requests.get(url, headers={"Authorization": PEXELS_API_KEY}, timeout=15)
        if r.status_code != 200:
            raise Exception(f"Pexels returned {r.status_code}")
        photos = r.json().get("photos", [])
        if not photos:
            raise Exception("No photos returned")
        portrait_photos = [p for p in photos if p["height"] >= p["width"] * 0.8]
        pool   = portrait_photos if portrait_photos else photos
        chosen = random.choice(pool)
        print(f"    ✅ Real cake URL found! ({chosen['width']}x{chosen['height']})")
        return chosen["src"]["large"]
    except Exception as e:
        print(f"    ❌ Pexels failed: {e} — using backup")
        return random.choice(REAL_BACKUPS)


@app.get("/api/get-cakes")
def get_cakes():
    theme    = pick_theme()
    print(f"\n🎂 ROUND THEME: {theme}")
    real_url = get_real_cake_url(theme)
    print(f"[2] Downloading image...")
    try:
        img = download_image(real_url)
        print(f"    ✅ Downloaded ({img.size[0]}x{img.size[1]}px)")
        print(f"[3] Detecting cake region and applying change...")
        fake_img, cx_pct, cy_pct, r_pct = apply_subtle_change(img)
        fake_b64 = pil_to_base64(fake_img)
        print(f"    ✅ Done! Diff at ({cx_pct},{cy_pct}) r={r_pct}")
        return {
            "real":  real_url,
            "fake":  fake_b64,
            "theme": theme,
            "diff":  {"cx": cx_pct, "cy": cy_pct, "r": r_pct, "w": img.size[0], "h": img.size[1]}
        }
    except Exception as e:
        print(f"    ❌ Processing failed: {e}")
        return {
            "real":  random.choice(REAL_BACKUPS),
            "fake":  random.choice(REAL_BACKUPS),
            "theme": theme,
            "diff":  {"cx": 0.5, "cy": 0.5, "r": 0.08, "w": 800, "h": 600}
        }


@app.get("/")
def root():
    return {"status": "ok", "message": "Is it Cake — running on Vercel!"}
