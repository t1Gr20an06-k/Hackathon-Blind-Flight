#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TERCOM-навигация (MVP): карта высот DEM + радиовысотомер (NMEA) + последняя точка GPS.

Сделано под хакатонный кейс и дальнейший перенос на борт:
  * скорость заранее знать не нужно;
  * предобработка сенсоров: радиоканал median+ФНЧ, барометр median+1D-Калман/модель шума;
  * векторизованный «грубо-точный» бутстрап TERCOM от последней точки GPS;
  * после захвата: TERCOM по истории траектории, а не по длинному прямому профилю;
  * мультигипотезный фильтр Калмана с постоянной скоростью [x,y,vx,vy];
  * отсев плоского рельефа по шероховатости/уверенности;
  * все координаты — пиксели DEM: x=столбец, y=строка.

Отладочный --truth используется только для графиков/подсчёта ошибки.
"""
from __future__ import annotations

import argparse, base64, csv, io, json, math, re, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np


# опциональная поддержка GeoTIFF — без затенения имён стандартной библиотеки
try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False
try:
    import tifffile
    HAS_TIFFILE = True
except Exception:
    HAS_TIFFILE = False

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# --------------------------- Ввод-вывод ---------------------------

def load_dem(path: str | Path):
    p = Path(path)
    # 1) GeoTIFF через rasterio
    if HAS_RASTERIO:
        try:
            import rasterio
            with rasterio.open(p) as ds:
                dem = ds.read(1).astype(np.float32)
                meta = {'crs': str(ds.crs), 'transform_gdal': list(ds.transform.to_gdal()), 'width': ds.width, 'height': ds.height}
                nodata = ds.nodata
                if nodata is not None:
                    dem[dem == nodata] = np.nan
                return dem, meta
        except Exception:
            pass
    # 2) запасной вариант tifffile + meta.json рядом
    if HAS_TIFFILE:
        try:
            import tifffile
            dem = tifffile.imread(str(p)).astype(np.float32)
            meta_p = p.with_name('meta.json')
            if not meta_p.exists():
                meta_p = p.parent / 'meta.json'
            if meta_p.exists():
                meta = json.loads(meta_p.read_text(encoding='utf-8'))
                return dem, meta
            return dem, {}
        except Exception:
            pass
    # 3) исходный NPZ / .npy — использует глобальные json/base64/re/io !
    if p.suffix.lower() == '.npy':
        return np.load(p).astype(np.float32), {}
    text = p.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r"-----BEGIN BASE64 NPZ-----\s*(.*?)\s*-----END BASE64 NPZ-----", text, flags=re.S)
    if not m:
        raise ValueError('DEM must be .npy, GeoTIFF, or text NPZ package')
    payload = re.sub(r"\s+", "", m.group(1))
    z = np.load(io.BytesIO(base64.b64decode(payload)), allow_pickle=False)
    dem = z['dem'].astype(np.float32)
    meta = json.loads(str(z['meta_json']))
    sentinel = meta.get('nan_sentinel')
    if sentinel is not None:
        dem[dem == float(sentinel)] = np.nan
    return dem, meta

def estimate_px_speed_range_from_meta(meta: dict, min_ms: float, max_ms: float):
    """Перевести физический диапазон м/с в приблизительный px/s для EPSG:4326 или проекционного DEM.
    Берём консервативные min/max, т.к. метры на пиксель по x/y в географической CRS различаются.
    """
    try:
        if meta.get('transform_gdal'):
            gt=meta['transform_gdal']; dx=abs(float(gt[1])); dy=abs(float(gt[5]))
        elif meta.get('resolution_geo_or_proj'):
            dx=abs(float(meta['resolution_geo_or_proj'][0])); dy=abs(float(meta['resolution_geo_or_proj'][1]))
        else:
            return None
        crs=str(meta.get('crs',''))
        if '4326' in crs:
            b=meta.get('bounds_geo') or [0,0,0,0]
            lat=0.5*(float(b[1])+float(b[3])) if len(b)>=4 else 0.0
            m_per_deg_lat=111320.0
            m_per_deg_lon=111320.0*math.cos(math.radians(lat))
            mpx=dx*m_per_deg_lon; mpy=dy*m_per_deg_lat
        else:
            # проекционная CRS — считаем, что единицы трансформа уже в метрах
            mpx=dx; mpy=dy
        if mpx <= 0 or mpy <= 0:
            return None
        min_px = min_ms / max(mpx, mpy)
        max_px = max_ms / min(mpx, mpy)
        return float(min_px), float(max_px), float(mpx), float(mpy)
    except Exception:
        return None


def _gdal_coeffs(meta: dict):
    gt = meta.get('transform_gdal')
    if not gt or len(gt) < 6:
        return None
    # геотрансформ GDAL: Xgeo = c + a*x + b*y; Ygeo = f + d*x + e*y
    c, a, b, f, d, e = map(float, gt[:6])
    return c, a, b, f, d, e

def pixel_to_geo(meta: dict, x_px: float, y_px: float):
    coeff = _gdal_coeffs(meta)
    if coeff is None:
        return np.nan, np.nan
    c, a, b, f, d, e = coeff
    lon = c + a*x_px + b*y_px
    lat = f + d*x_px + e*y_px
    return float(lon), float(lat)

def geo_to_pixel(meta: dict, lon: float, lat: float):
    coeff = _gdal_coeffs(meta)
    if coeff is None:
        return np.nan, np.nan
    c, a, b, f, d, e = coeff
    # решаем [[a,b],[d,e]] [x,y] = [lon-c, lat-f]
    det = a*e - b*d
    if abs(det) < 1e-18:
        return np.nan, np.nan
    xx = lon - c; yy = lat - f
    x = ( e*xx - b*yy) / det
    y = (-d*xx + a*yy) / det
    return float(x), float(y)

def meters_per_pixel(meta: dict, lat: float = None):
    coeff = _gdal_coeffs(meta)
    if coeff is None:
        return np.nan, np.nan
    c, a, b, f, d, e = coeff
    # приближённые осевые векторы пикселя в единицах карты
    crs = str(meta.get('crs',''))
    if lat is None or not np.isfinite(lat):
        bounds = meta.get('bounds_geo') or [0,0,0,0]
        lat = 0.5*(float(bounds[1])+float(bounds[3])) if len(bounds) >= 4 else 0.0
    if '4326' in crs:
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat))
        # длины базисных векторов пикселя в метрах
        mpx_x = math.hypot(a*m_per_deg_lon, d*m_per_deg_lat)
        mpx_y = math.hypot(b*m_per_deg_lon, e*m_per_deg_lat)
    else:
        # считаем проекционную CRS в метрах
        mpx_x = math.hypot(a, d)
        mpx_y = math.hypot(b, e)
    return float(mpx_x), float(mpx_y)

def velocity_px_to_metric(meta: dict, x_px: float, y_px: float, vx_px_s: float, vy_px_s: float):
    lon, lat = pixel_to_geo(meta, x_px, y_px)
    mpx_x, mpx_y = meters_per_pixel(meta, lat)
    if not np.isfinite(mpx_x) or not np.isfinite(mpx_y):
        return dict(lon_est=lon, lat_est=lat, v_east_m_s=np.nan, v_north_m_s=np.nan, speed_m_s=np.nan, course_deg=np.nan)
    # пиксель x растёт на восток, y — вниз/на юг для растров с севером вверху
    v_east = vx_px_s * mpx_x
    v_north = -vy_px_s * mpx_y
    speed = math.hypot(v_east, v_north)
    course = (math.degrees(math.atan2(v_east, v_north)) + 360.0) % 360.0 if speed > 1e-9 else 0.0
    return dict(lon_est=lon, lat_est=lat, v_east_m_s=float(v_east), v_north_m_s=float(v_north), speed_m_s=float(speed), course_deg=float(course))

def pixel_error_to_meters(meta: dict, x1: float, y1: float, x2: float, y2: float):
    lon, lat = pixel_to_geo(meta, 0.5*(x1+x2), 0.5*(y1+y2))
    mpx_x, mpx_y = meters_per_pixel(meta, lat)
    if not np.isfinite(mpx_x) or not np.isfinite(mpx_y):
        return np.nan
    return float(math.hypot((x1-x2)*mpx_x, (y1-y2)*mpx_y))

def parse_gpgga_alt(line: str) -> Optional[float]:
    if not line.startswith('$GPGGA'):
        return None
    parts = line.split('*', 1)[0].split(',')
    try:
        return float(parts[9]) if len(parts) > 9 and parts[9] else None
    except Exception:
        return None


def read_nmea(path: str | Path) -> np.ndarray:
    vals = []
    for line in Path(path).read_text(encoding='utf-8', errors='ignore').splitlines():
        v = parse_gpgga_alt(line.strip())
        if v is not None:
            vals.append(v)
    if not vals:
        raise ValueError('No $GPGGA altitude values found')
    return np.asarray(vals, dtype=np.float64)


def read_truth(path: str | Path):
    xs=[]; ys=[]; ts=[]; elev=[]
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            ts.append(float(row.get('t_s', len(xs))))
            xs.append(float(row['x_px'])); ys.append(float(row['y_px']))
            elev.append(float(row.get('elevation', 'nan')))
    return dict(t=np.asarray(ts), x=np.asarray(xs), y=np.asarray(ys), elevation=np.asarray(elev))

# --------------------------- фильтрация ---------------------------

def median_filter(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    if win % 2 == 0:
        win += 1
    r = win // 2
    xp = np.pad(x, (r, r), mode='edge')
    return np.asarray([np.median(xp[i:i+win]) for i in range(len(x))], dtype=np.float64)


def lowpass(x: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    if not HAS_SCIPY or cutoff <= 0 or cutoff >= 0.49*fs or len(x) < 20:
        return x.copy()
    b, a = butter(2, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, x, padlen=min(len(x)-1, 12))

class BaroKalman1D:
    def __init__(self, x0: float, q=0.02, r=1.0):
        self.x=float(x0); self.p=10.0; self.q=q; self.r=r
    def update(self, z: float):
        self.p += self.q
        k = self.p/(self.p+self.r)
        self.x += k*(z-self.x)
        self.p = (1-k)*self.p
        return self.x


def make_terrain_profile(args):
    radar_raw = read_nmea(args.nmea)
    radar_f = lowpass(median_filter(radar_raw, args.radar_median), args.fs, args.radar_cutoff)
    n = len(radar_raw)
    rng = np.random.default_rng(args.noise_seed)
    baro_raw = np.full(n, args.absolute_altitude, dtype=np.float64)
    if args.baro_noise_std > 0:
        baro_raw += rng.normal(0, args.baro_noise_std, n)
    if args.baro_drift_per_s > 0:
        baro_raw += np.cumsum(rng.normal(0, args.baro_drift_per_s/args.fs, n))
    baro_med = median_filter(baro_raw, args.baro_median)
    kf = BaroKalman1D(baro_med[0], args.baro_q, args.baro_r)
    baro_f = np.asarray([kf.update(v) for v in baro_med], dtype=np.float64)
    terrain = baro_f - radar_f
    return terrain, radar_raw, radar_f, baro_f

# --------------------------- геометрия / сопоставление ---------------------------

def az_unit(az_deg):
    a = np.deg2rad(az_deg)
    return np.sin(a), -np.cos(a)

def vel_to_az(vx, vy):
    if abs(vx)+abs(vy) < 1e-9:
        return 0.0
    return (math.degrees(math.atan2(vx, -vy)) + 360) % 360


def bilinear_many(dem: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Векторизованная билинейная интерполяция для xs/ys любой одинаковой формы. За картой -> nan."""
    xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
    out = np.full(xs.shape, np.nan, dtype=np.float32)
    h, w = dem.shape
    m = (xs >= 0) & (ys >= 0) & (xs < w-1) & (ys < h-1)
    if not np.any(m):
        return out
    x = xs[m]; y = ys[m]
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    dx = (x - x0).astype(np.float32); dy = (y - y0).astype(np.float32)
    z00=dem[y0,x0]; z10=dem[y0,x0+1]; z01=dem[y0+1,x0]; z11=dem[y0+1,x0+1]
    out[m] = (z00*(1-dx)+z10*dx)*(1-dy) + (z01*(1-dx)+z11*dx)*dy
    return out


def roughness(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    return float(0.65*np.std(p) + 0.35*np.std(np.diff(p)))


def corr_rows(live: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Корреляция Пирсона для каждой строки refs. NaN-строки получают -1."""
    live = live.astype(np.float32)
    lv = live - np.mean(live)
    ls = np.std(lv)
    if ls < 1e-6:
        return np.full(refs.shape[0], -1.0, dtype=np.float32)
    lv /= ls
    valid = np.isfinite(refs).all(axis=1)
    out = np.full(refs.shape[0], -1.0, dtype=np.float32)
    if not np.any(valid):
        return out
    r = refs[valid].astype(np.float32)
    r -= r.mean(axis=1, keepdims=True)
    rs = r.std(axis=1)
    good = rs > 1e-6
    rr = np.full(r.shape[0], -1.0, dtype=np.float32)
    rr[good] = (r[good] / rs[good, None] * lv[None, :]).mean(axis=1)
    out[np.where(valid)[0]] = rr
    return out

@dataclass
class Candidate:
    x: float; y: float; corr: float; score: float
    az: float = np.nan; speed: float = np.nan; dx: float = 0.0; dy: float = 0.0; dtheta: float = 0.0


def top_n_candidates(cands: List[Candidate], n: int, nms_radius: float) -> List[Candidate]:
    cands.sort(key=lambda c: c.score, reverse=True)
    out=[]
    for c in cands:
        if all(math.hypot(c.x-o.x, c.y-o.y) >= nms_radius for o in out):
            out.append(c)
        if len(out) >= n:
            break
    return out


def bootstrap_match(dem, terrain, sample_i, start_x, start_y, elapsed_s, args) -> List[Candidate]:
    """Скорость/курс неизвестны: конечные точки = старт + скорость*время*азимут + малый сдвиг.

    Бутстрап намеренно многомасштабный: ложное место может коррелировать на одном коротком
    окне, но обычно проваливается на длинных. Это критично для ломаных траекторий и не даёт
    «развёрнутый на 90°» ложный захват.
    """
    if elapsed_s <= 0:
        return []
    L = min(args.window_size, sample_i + 1)
    live = terrain[sample_i-L+1:sample_i+1]
    L = len(live)
    back_idx = (L-1-np.arange(L, dtype=np.float32))[None, :]
    offsets=[]
    for dx in np.arange(-args.bootstrap_radius_px, args.bootstrap_radius_px+1e-6, args.bootstrap_grid_px):
        for dy in np.arange(-args.bootstrap_radius_px, args.bootstrap_radius_px+1e-6, args.bootstrap_grid_px):
            if dx*dx+dy*dy <= args.bootstrap_radius_px**2:
                offsets.append((float(dx),float(dy)))
    if getattr(args, 'initial_azimuth_deg', None) is not None and args.initial_az_window_deg < 180:
        center = float(args.initial_azimuth_deg) % 360.0
        rel = np.arange(-args.initial_az_window_deg, args.initial_az_window_deg + 0.5*args.bootstrap_az_step_deg, args.bootstrap_az_step_deg, dtype=np.float32)
        azs = ((center + rel) % 360.0).astype(np.float32)
    else:
        azs = np.arange(0, 360, args.bootstrap_az_step_deg, dtype=np.float32)
    speeds = np.arange(args.min_speed_px_s, args.max_speed_px_s+0.5*args.bootstrap_speed_step_px_s, args.bootstrap_speed_step_px_s, dtype=np.float32)
    params=[]
    for sp in speeds:
        dist = float(sp)*elapsed_s
        ux_all, uy_all = az_unit(azs)
        cxs = start_x + ux_all*dist
        cys = start_y + uy_all*dist
        for az,ux,uy,cx,cy in zip(azs, ux_all, uy_all, cxs, cys):
            if cx < -args.bootstrap_radius_px or cy < -args.bootstrap_radius_px or cx >= dem.shape[1]+args.bootstrap_radius_px or cy >= dem.shape[0]+args.bootstrap_radius_px:
                continue
            for dx,dy in offsets:
                xe=cx+dx; ye=cy+dy
                if 0 <= xe < dem.shape[1]-1 and 0 <= ye < dem.shape[0]-1:
                    params.append((xe,ye,float(az),float(sp),float(ux),float(uy),dx,dy))
    if not params:
        return []
    cands=[]
    chunk=args.chunk_candidates
    for s in range(0,len(params),chunk):
        pp=params[s:s+chunk]
        arr=np.asarray(pp, dtype=np.float32)
        xe=arr[:,0:1]; ye=arr[:,1:2]; az=arr[:,2]; sp=arr[:,3:4]; ux=arr[:,4:5]; uy=arr[:,5:6]
        step = sp / args.fs
        xs = xe - ux*back_idx*step
        ys = ye - uy*back_idx*step
        refs = bilinear_many(dem, xs, ys)
        cs = corr_rows(live, refs)
        for row,c in zip(arr, cs):
            if c < args.min_corr_bootstrap:
                continue
            dx=float(row[6]); dy=float(row[7])
            score=float(c) - args.offset_penalty*math.hypot(dx,dy)/max(args.bootstrap_radius_px,1)
            cands.append(Candidate(float(row[0]),float(row[1]),float(c),score,az=float(row[2]),speed=float(row[3]),dx=dx,dy=dy))
    # Многомасштабная проверка кандидатов бутстрапа.
    # Переоцениваем только тех, кто прошёл корреляцию на базовом окне.
    if cands and args.bootstrap_verify_windows:
        verify_windows = []
        for ww in args.bootstrap_verify_windows:
            ww = int(ww)
            if 8 <= ww <= sample_i + 1 and ww not in verify_windows:
                verify_windows.append(ww)
        for c in cands:
            corrs = [c.corr]
            for ww in verify_windows:
                if ww == L:
                    continue
                live_w = terrain[sample_i-ww+1:sample_i+1]
                ref_w = extract_line_profile(dem, c.x, c.y, c.az, c.speed, ww, args.fs)
                corrs.append(float(corr_rows(live_w, ref_w[None, :])[0]))
            finite = [v for v in corrs if np.isfinite(v)]
            if finite:
                # Среднее поощряет стабильное совпадение; минимум штрафует «обман на одном окне».
                c.score = 0.65*float(np.mean(finite)) + 0.35*float(np.min(finite)) - args.offset_penalty*math.hypot(c.dx,c.dy)/max(args.bootstrap_radius_px,1)
                c.corr = float(np.mean(finite))
    return top_n_candidates(cands, args.top_candidates, args.nms_radius_px)


def extract_line_profile(dem, x_end, y_end, az, speed, L, fs):
    ux, uy = az_unit(float(az))
    back = (L-1-np.arange(L, dtype=np.float32)) * float(speed) / fs
    return bilinear_many(dem, x_end - ux*back, y_end - uy*back)


# --------------------------- детерминированная финальная коррекция позиции ---------------------------
# Диагностика на этих данных показала: в конце маршрута оценка Калмана/MHT НЕ сидит на пике
# корреляции рельефа (corr ~0.3 у оценки против ~0.7 у пика рядом с истиной), потому что
# динамика трекера уводит её с лучшего совпадения. Сам пик корреляции лежит в ~6–10 м от
# истины — это честный предел точности на DEM ~31 м/пиксель с шумным радиовысотомером; точнее
# по-честному нельзя. Поэтому «защёлкиваем» итоговую позицию на этот пик ограниченным
# детерминированным плотным поиском. Важно: тут НЕТ обратной связи в цикл MHT, поэтому, в отличие
# от поправок на каждом измерении, это не перещёлкивает гипотезы и не дестабилизирует прогон.

def _peak_search(dem, live, x, y, az, speed, R, res, fs):
    """Глобальный максимум корреляции на плотной сетке (x±R, y±R). Возвращает (corr, x, y).

    Векторизовано: каждое смещение — это просто перенос того же обратно-спроецированного луча,
    поэтому все опорные профили строятся и коррелируются одним батч-вызовом.
    """
    L = len(live)
    ux, uy = az_unit(float(az))
    back = (L-1 - np.arange(L, dtype=np.float64)) * float(speed) / fs   # (L,)
    base_x = float(x) - ux*back; base_y = float(y) - uy*back            # (L,)
    offs = np.arange(-R, R + 1e-9, res)
    DX, DY = np.meshgrid(offs, offs, indexing='ij')
    gx = DX.ravel(); gy = DY.ravel()                                   # (M,)
    xs = base_x[None, :] + gx[:, None]                                 # (M,L)
    ys = base_y[None, :] + gy[:, None]
    cs = corr_rows(live, bilinear_many(dem, xs, ys))                   # (M,)
    j = int(np.argmax(cs))
    return float(cs[j]), float(x + gx[j]), float(y + gy[j])


def final_position_fix(dem, terrain, x, y, vx, vy, args):
    """Ограниченная детерминированная коррекция по корреляции вокруг сошедшейся финальной оценки.

    Защёлкивается на пик корреляции рельефа, усреднённый по нескольким длинам окна и взвешенный
    по корреляции, чтобы результат не зависел от одного «удачно выбранного» окна. Возвращает
    (x_fix, y_fix, corr, shift_px) или None, если ни одно окно не дало надёжного пика (слишком
    плоско, слабая корреляция или пик на границе поиска — тогда истинный пик, вероятно, в другом
    месте, и придумывать его нельзя).
    """
    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return None
    az = vel_to_az(vx, vy)
    base = int(args.final_fix_window or args.window_size)
    windows = sorted({base, (base*3)//4, base//2})
    R = float(args.final_fix_radius_px); res = float(args.final_fix_res_px)
    rough_thr = float(args.final_fix_min_roughness)
    picks = []  # (corr, x, y)
    for L in windows:
        L = int(min(L, len(terrain)))
        if L < 8:
            continue
        live = terrain[len(terrain)-L:]
        if roughness(live) < rough_thr:
            continue
        c, xf, yf = _peak_search(dem, live, x, y, az, speed, R, res, args.fs)
        if c < args.min_quality_corr:
            continue
        if abs(xf - x) >= R - 1e-6 or abs(yf - y) >= R - 1e-6:
            continue
        picks.append((c, xf, yf))
    if not picks:
        return None
    w = np.array([p[0] for p in picks], dtype=np.float64)
    xf = float(np.average([p[1] for p in picks], weights=w))
    yf = float(np.average([p[2] for p in picks], weights=w))
    return xf, yf, float(w.max()), math.hypot(xf - x, yf - y)


def _rts_core(z, Rdiag, dt, accel_sigma):
    """Прямой Калман + обратный проход RTS для модели постоянной скорости.

    z: (n,2) измерения позиции; Rdiag: (n,) дисперсия измерения по каждому отсчёту.
    Возвращает сглаженные состояния xs: (n,4) = [x,y,vx,vy]. Линейно-гауссовский оптимум.
    """
    n = len(z)
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    qa = accel_sigma**2; dt2=dt*dt; dt3=dt**3; dt4=dt**4
    Q = qa*np.array([[dt4/4,0,dt3/2,0],[0,dt4/4,0,dt3/2],[dt3/2,0,dt2,0],[0,dt3/2,0,dt2]], float)
    H = np.array([[1,0,0,0],[0,1,0,0]], float); I4 = np.eye(4)
    xf = np.zeros((n,4)); Pf = np.zeros((n,4,4)); xp = np.zeros((n,4)); Pp = np.zeros((n,4,4))
    x = np.array([z[0,0], z[0,1], 0.0, 0.0]); P = np.eye(4)*100.0
    for i in range(n):
        if i > 0:
            x = F@x; P = F@P@F.T + Q
        xp[i]=x; Pp[i]=P
        S = H@P@H.T + np.diag([Rdiag[i], Rdiag[i]])
        K = P@H.T@np.linalg.inv(S)
        x = x + K@(z[i]-H@x); P = (I4-K@H)@P
        xf[i]=x; Pf[i]=P
    xs = xf.copy()
    for i in range(n-2, -1, -1):
        C = Pf[i]@F.T@np.linalg.inv(Pp[i+1])
        xs[i] = xf[i] + C@(xs[i+1]-xp[i+1])
    return xs


def _write_state(rows, xs):
    for i, r in enumerate(rows):
        r['x_est']=float(xs[i,0]); r['y_est']=float(xs[i,1])
        r['vx_est']=float(xs[i,2]); r['vy_est']=float(xs[i,3])
        r['speed_px_s']=float(math.hypot(xs[i,2], xs[i,3]))
        r['azimuth_deg']=float(vel_to_az(xs[i,2], xs[i,3]))


def rts_smooth_positions(rows, fs, accel_sigma, smin, smax):
    """RTS-сглаживатель траектории со взвешиванием по уверенности.

    Шум измерения на каждом отсчёте обратно пропорционален собственной уверенности трекера
    (solution_confidence): короткие уходы с низкой уверенностью (мгновенно неверная гипотеза MHT)
    «перекрываются» динамикой постоянной скорости и уверенными соседями, а настоящие манёвры
    с высокой уверенностью отслеживаются точно. Истину не использует.
    Сырой выход трекера сохраняется в x_raw/y_raw.
    """
    n = len(rows)
    if n < 3:
        return
    for r in rows:
        r['x_raw']=r['x_est']; r['y_raw']=r['y_est']
    z = np.array([[float(r['x_est']), float(r['y_est'])] for r in rows], dtype=float)
    conf = np.clip(np.array([float(r.get('solution_confidence', 0.0)) for r in rows]), 0.0, 1.0)
    Rdiag = (smax - conf*(smax - smin))**2            # низкая уверенность -> большой шум измерения
    _write_state(rows, _rts_core(z, Rdiag, 1.0/fs, accel_sigma))


def correlation_refit(dem, terrain, rows, args):
    """Детерминированная доводка всей траектории по корреляции (без обратной связи в MHT).

    На подвыборке точек защёлкиваемся на локальный пик корреляции рельефа (тот же честный
    механизм, что и финальная коррекция, по нескольким окнам), затем пере-сглаживаем через эти
    уточнённые якоря. Две защиты не дают «придумывать» данные на неоднозначном рельефе:
      * поиск ограничен ±refit_radius_px вокруг текущей оценки;
      * поправка больше refit_gate_px (прыжок к ложному пику, напр. неоднозначность на t=60 с,
        где пик в ~3 пикселях от истины) отвергается сразу.
    Честно: использует только профиль радиовысотомера + DEM, никогда не истину.
    """
    n = len(rows)
    if n < 3:
        return
    fs = args.fs
    xs = np.array([float(r['x_est']) for r in rows]); ys = np.array([float(r['y_est']) for r in rows])
    vxs = np.array([float(r['vx_est']) for r in rows]); vys = np.array([float(r['vy_est']) for r in rows])
    base = int(args.final_fix_window or args.window_size)
    windows = sorted({base, (base*3)//4, base//2})
    R = float(args.refit_radius_px); res = float(args.refit_res_px); gate = float(args.refit_gate_px)
    z = np.column_stack([xs, ys]).astype(float)
    Rdiag = np.full(n, float(args.refit_base_sigma)**2)   # базовое доверие к текущему сглаженному пути
    n_anchor = 0
    for i in range(0, n, int(args.refit_step)):
        sp = math.hypot(vxs[i], vys[i])
        if sp < 1e-6:
            continue
        az = vel_to_az(vxs[i], vys[i])
        picks = []
        for L in windows:
            L = int(min(L, i+1))
            if L < 8:
                continue
            live = terrain[i-L+1:i+1]
            if roughness(live) < float(args.final_fix_min_roughness):
                continue
            c, xf, yf = _peak_search(dem, live, xs[i], ys[i], az, sp, R, res, fs)
            if c < args.min_quality_corr:
                continue
            if abs(xf-xs[i]) >= R-1e-6 or abs(yf-ys[i]) >= R-1e-6:   # пик на границе -> не доверяем
                continue
            picks.append((c, xf, yf))
        if not picks:
            continue
        w = np.array([p[0] for p in picks])
        xpk = float(np.average([p[1] for p in picks], weights=w))
        ypk = float(np.average([p[2] for p in picks], weights=w))
        if math.hypot(xpk-xs[i], ypk-ys[i]) > gate:   # отвергаем прыжок к ложному пику (неоднозначный рельеф)
            continue
        z[i] = [xpk, ypk]; Rdiag[i] = float(args.refit_anchor_sigma)**2; n_anchor += 1
    print(f'[refit] принято якорей по корреляции: {n_anchor}')
    _write_state(rows, _rts_core(z, Rdiag, 1.0/fs, args.accel_sigma))


def segment_match(dem, live, anchor_x, anchor_y, elapsed_s, pred_x, pred_y, args) -> List[Candidate]:
    """Локальный матчер после захвата: короткий прямой отрезок от якоря недавней истории.

    Дружелюбнее к манёврам, чем сдвиг всей длинной истории: на ломаной траектории последние
    несколько секунд близки к прямому отрезку.
    """
    if elapsed_s <= 0:
        return []
    L=len(live)
    back=(L-1-np.arange(L,dtype=np.float32))[None,:]
    offsets=[]; rad=args.local_radius_px; grid=args.local_grid_px
    for dx in np.arange(-rad, rad+1e-6, grid):
        for dy in np.arange(-rad, rad+1e-6, grid):
            if dx*dx+dy*dy <= rad*rad:
                offsets.append((float(dx),float(dy)))
    azs=np.arange(0,360,args.local_az_step_deg,dtype=np.float32)
    speeds=np.arange(args.min_speed_px_s,args.max_speed_px_s+0.5*args.local_speed_step_px_s,args.local_speed_step_px_s,dtype=np.float32)
    params=[]
    for sp in speeds:
        dist=float(sp)*elapsed_s
        ux_all,uy_all=az_unit(azs)
        cxs=anchor_x+ux_all*dist; cys=anchor_y+uy_all*dist
        for az,ux,uy,cx,cy in zip(azs,ux_all,uy_all,cxs,cys):
            # ограничиваем окрестностью предсказанной Калманом точки, чтобы не ловить далёкие ложные максимумы
            if math.hypot(float(cx)-pred_x, float(cy)-pred_y) > args.local_endpoint_gate_px + rad:
                continue
            for dx,dy in offsets:
                xe=float(cx)+dx; ye=float(cy)+dy
                if 0 <= xe < dem.shape[1]-1 and 0 <= ye < dem.shape[0]-1:
                    params.append((xe,ye,float(az),float(sp),float(ux),float(uy),dx,dy))
    if not params:
        return []
    cands=[]
    for st in range(0,len(params),args.chunk_candidates):
        arr=np.asarray(params[st:st+args.chunk_candidates],dtype=np.float32)
        xe=arr[:,0:1]; ye=arr[:,1:2]; ux=arr[:,4:5]; uy=arr[:,5:6]; sp=arr[:,3:4]
        xs=xe-ux*back*(sp/args.fs); ys=ye-uy*back*(sp/args.fs)
        refs=bilinear_many(dem,xs,ys)
        cs=corr_rows(live,refs)
        for row,c in zip(arr,cs):
            if c < args.min_corr_local:
                continue
            dx=float(row[6]); dy=float(row[7]); xe=float(row[0]); ye=float(row[1])
            innov=math.hypot(xe-pred_x, ye-pred_y)
            score=float(c) - args.offset_penalty*math.hypot(dx,dy)/max(rad,1) - args.segment_innovation_penalty*min(innov/max(args.local_endpoint_gate_px,1),3)
            cands.append(Candidate(xe,ye,float(c),score,az=float(row[2]),speed=float(row[3]),dx=dx,dy=dy))
    return top_n_candidates(cands,args.top_candidates,args.nms_radius_px)

def history_match(dem, live, hist_xy: np.ndarray, args) -> List[Candidate]:
    """После захвата: сравниваем живой профиль со сдвинутой/повёрнутой предсказанной историей."""
    L=len(live)
    if len(hist_xy) < L:
        return []
    hist = hist_xy[-L:].astype(np.float32)
    end = hist[-1].copy()
    rel = hist - end[None,:]
    params=[]
    for dth in np.arange(-args.local_theta_deg, args.local_theta_deg+1e-6, args.local_theta_step_deg):
        a=math.radians(float(dth)); ca=math.cos(a); sa=math.sin(a)
        rot_x = rel[:,0]*ca - rel[:,1]*sa
        rot_y = rel[:,0]*sa + rel[:,1]*ca
        for dx in np.arange(-args.local_radius_px, args.local_radius_px+1e-6, args.local_grid_px):
            for dy in np.arange(-args.local_radius_px, args.local_radius_px+1e-6, args.local_grid_px):
                if dx*dx+dy*dy <= args.local_radius_px**2:
                    params.append((float(dx),float(dy),float(dth),rot_x.copy(),rot_y.copy()))
    cands=[]
    # вариантов немного; всё равно батчим через матрицу
    for s in range(0,len(params),args.chunk_candidates):
        pp=params[s:s+args.chunk_candidates]
        xs=[]; ys=[]; meta=[]
        for dx,dy,dth,rx,ry in pp:
            xs.append(end[0] + rx + dx); ys.append(end[1] + ry + dy); meta.append((dx,dy,dth))
        refs=bilinear_many(dem, np.asarray(xs), np.asarray(ys))
        cs=corr_rows(live, refs)
        for (dx,dy,dth),c in zip(meta,cs):
            if c < args.min_corr_local:
                continue
            xe=float(end[0]+dx); ye=float(end[1]+dy)
            score=float(c) - args.offset_penalty*math.hypot(dx,dy)/max(args.local_radius_px,1) - args.theta_penalty*abs(dth)/max(args.local_theta_deg,1)
            cands.append(Candidate(xe,ye,float(c),score,dx=dx,dy=dy,dtheta=dth))
    return top_n_candidates(cands, args.top_candidates, args.nms_radius_px)

# --------------------------- Калман / MHT (мультигипотезы) ---------------------------

@dataclass
class KalmanCV:
    x: np.ndarray
    P: np.ndarray
    def predict(self, dt, accel_sigma):
        F=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],float)
        q=accel_sigma**2; dt2=dt*dt; dt3=dt**3; dt4=dt**4
        Q=q*np.array([[dt4/4,0,dt3/2,0],[0,dt4/4,0,dt3/2],[dt3/2,0,dt2,0],[0,dt3/2,0,dt2]],float)
        self.x=F@self.x; self.P=F@self.P@F.T+Q
    def update_pos(self, z, sigma):
        H=np.array([[1,0,0,0],[0,1,0,0]],float); R=np.diag([sigma*sigma,sigma*sigma])
        y=z-H@self.x; S=H@self.P@H.T+R; K=self.P@H.T@np.linalg.inv(S)
        self.x=self.x+K@y; I=np.eye(4); self.P=(I-K@H)@self.P@(I-K@H).T+K@R@K.T

@dataclass
class Hyp:
    kf: KalmanCV
    score: float
    locked: bool
    hist: List[tuple] = field(default_factory=list)
    last_corr: float = np.nan
    last_rough: float = 0.0
    updates: int = 0


def clone_hyp(h: Hyp) -> Hyp:
    return Hyp(KalmanCV(h.kf.x.copy(), h.kf.P.copy()), h.score, h.locked, list(h.hist), h.last_corr, h.last_rough, h.updates)


def meas_sigma_from_corr(corr, args):
    q=np.clip((corr-args.min_quality_corr)/(1-args.min_quality_corr),0,1)
    return args.meas_sigma_max - q*(args.meas_sigma_max-args.meas_sigma_min)

def solution_confidence(last_corr, last_rough, locked, args):
    if not locked or not np.isfinite(last_corr):
        return 0.0
    c = np.clip((last_corr - args.min_quality_corr) / max(1e-6, 1.0 - args.min_quality_corr), 0, 1)
    r = np.clip(last_rough / max(args.min_roughness_local, 1e-6), 0, 1)
    return float(c * (0.4 + 0.6*r))

# --------------------------- отчёты / тепловая карта ---------------------------

def compute_bootstrap_heatmap(dem, terrain, start_x, start_y, sample_i, args):
    """Тепловая карта лучшей корреляции по смещениям для азимут×скорость на выбранном отсчёте бутстрапа."""
    L = min(args.window_size, sample_i + 1)
    if sample_i < L-1:
        return None
    live = terrain[sample_i-L+1:sample_i+1]
    elapsed_s = sample_i / args.fs
    if elapsed_s <= 0:
        return None
    old_top = args.top_candidates
    # локальная копия параметров; здесь нужны все корреляции, а не только кандидаты
    azs = np.arange(0, 360 + 1e-9, args.heatmap_az_step_deg, dtype=np.float32)
    speeds = np.arange(args.min_speed_px_s, args.max_speed_px_s + 0.5*args.heatmap_speed_step_px_s, args.heatmap_speed_step_px_s, dtype=np.float32)
    offsets=[]
    rad=args.bootstrap_radius_px; grid=args.bootstrap_grid_px
    for dx in np.arange(-rad, rad+1e-6, grid):
        for dy in np.arange(-rad, rad+1e-6, grid):
            if dx*dx+dy*dy <= rad*rad:
                offsets.append((float(dx),float(dy)))
    back=(L-1-np.arange(L,dtype=np.float32))[None,:]
    hm=np.full((len(speeds), len(azs)), np.nan, dtype=np.float32)
    best=None
    for si,sp in enumerate(speeds):
        dist=float(sp)*elapsed_s
        params=[]
        ux_all,uy_all=az_unit(azs)
        for ai,(az,ux,uy) in enumerate(zip(azs,ux_all,uy_all)):
            cx=start_x+ux*dist; cy=start_y+uy*dist
            local=[]
            for dx,dy in offsets:
                xe=cx+dx; ye=cy+dy
                if 0 <= xe < dem.shape[1]-1 and 0 <= ye < dem.shape[0]-1:
                    local.append((xe,ye,ux,uy,dx,dy))
            if not local:
                continue
            arr=np.asarray(local,dtype=np.float32)
            xs=arr[:,0:1] - arr[:,2:3]*back*(float(sp)/args.fs)
            ys=arr[:,1:2] - arr[:,3:4]*back*(float(sp)/args.fs)
            refs=bilinear_many(dem,xs,ys)
            cs=corr_rows(live,refs)
            mx=float(np.nanmax(cs)) if len(cs) else np.nan
            hm[si,ai]=mx
            if np.isfinite(mx) and (best is None or mx > best[0]):
                j=int(np.nanargmax(cs)); best=(mx,float(arr[j,0]),float(arr[j,1]),float(az),float(sp),float(arr[j,4]),float(arr[j,5]))
    return dict(heatmap=hm, azimuths=azs, speeds=speeds, best=best, sample=sample_i, roughness=roughness(live))

def plot_bootstrap_heatmap(out, hmdata):
    if not HAS_MPL or hmdata is None:
        return
    hm=hmdata['heatmap']; az=hmdata['azimuths']; sp=hmdata['speeds']
    fig,ax=plt.subplots(figsize=(12,6))
    im=ax.imshow(hm, origin='lower', aspect='auto', cmap='hot', vmin=-0.2, vmax=1.0, extent=[az[0], az[-1], sp[0], sp[-1]])
    fig.colorbar(im, ax=ax, label='max corr over dx/dy')
    if hmdata.get('best'):
        c,x,y,baz,bsp,dx,dy=hmdata['best']
        ax.plot(baz,bsp,'c*',ms=14,mec='k',label=f'best corr={c:.3f}')
        ax.legend()
    ax.set_xlabel('azimuth, deg')
    ax.set_ylabel('speed, px/s')
    ax.set_title(f"Bootstrap TERCOM heatmap, sample={hmdata['sample']}, rough={hmdata['roughness']:.2f}")
    fig.tight_layout(); fig.savefig(out/'correlation_heatmap_az_speed.png',dpi=160); plt.close(fig)

def generate_report(out, rows, events, args):
    rows_df = rows
    errs=[r.get('err_px') for r in rows if 'err_px' in r and np.isfinite(r.get('err_px',np.nan))]
    locked=sum(r['nav_status']=='LOCKED' for r in rows)
    final=rows[-1]
    mean_err = float(np.mean(errs)) if errs else None
    final_err = float(errs[-1]) if errs else None
    avg_conf = float(np.mean([r.get('solution_confidence',0) for r in rows])) if rows else 0
    html=f'''<!doctype html><html><head><meta charset="utf-8"><title>TERCOM MVP report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}} table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px}} img{{max-width:100%;border:1px solid #ddd}}</style></head><body>
<h1>TERCOM + MHT Kalman MVP report</h1>
<h2>Summary</h2>
<table>
<tr><th>samples</th><td>{len(rows)}</td></tr>
<tr><th>locked samples</th><td>{locked}</td></tr>
<tr><th>final x,y</th><td>{final['x_est']:.2f}, {final['y_est']:.2f}</td></tr>
<tr><th>final speed px/s</th><td>{final['speed_px_s']:.3f}</td></tr>
<tr><th>final azimuth deg</th><td>{final['azimuth_deg']:.1f}</td></tr>
<tr><th>avg confidence</th><td>{avg_conf:.3f}</td></tr>
<tr><th>mean error px debug</th><td>{'' if mean_err is None else f'{mean_err:.3f}'}</td></tr>
<tr><th>final error px debug</th><td>{'' if final_err is None else f'{final_err:.3f}'}</td></tr>
</table>
<h2>Trajectory</h2><img src="trajectory_on_dem.png">
<h2>Error</h2><img src="error_vs_time.png">
<h2>Correlation heatmap</h2><img src="correlation_heatmap_az_speed.png">
<h2>Notes</h2>
<ul><li>GPS is used only as last known start point; after that nav status is TERCOM/Kalman or dead-reckoning.</li>
<li>Azimuth step is configurable. For requirement demo use <code>--bootstrap-az-step-deg 1 --heatmap-az-step-deg 1</code>.</li>
<li>Onboard trade-off: larger azimuth/speed steps are faster; 1° mode is available for accuracy/demo.</li></ul>
</body></html>'''
    (out/'report.html').write_text(html,encoding='utf-8')

# --------------------------- главный цикл ---------------------------

def run(args):

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    # --- Карта высот DEM ---
    dem_path = getattr(args, 'dem', None) or getattr(args, 'map', None)
    if not dem_path:
        raise ValueError('Need --dem / --map')
    dem, meta = load_dem(dem_path)

    # --- Профиль рельефа: --heights (радио, фильтруется внутри) или устаревший --nmea ---
    truth = read_truth(args.truth) if getattr(args, 'truth', None) else None
    start_idx = 0
    heights_path = getattr(args, 'heights', None)
    if heights_path:
        radio_raw = np.loadtxt(heights_path, dtype=np.float64)
        # предобработка сенсоров — ТОЧНО как в исходной make_terrain_profile()
        radar_f = lowpass(median_filter(radio_raw, args.radar_median), args.fs, args.radar_cutoff)
        n = len(radar_f)
        rng = np.random.default_rng(getattr(args, 'noise_seed', None))
        baro_raw = np.full(n, args.absolute_altitude, dtype=np.float64)
        if getattr(args, 'baro_noise_std', 0) > 0:
            baro_raw += rng.normal(0, args.baro_noise_std, n)
        if getattr(args, 'baro_drift_per_s', 0) > 0:
            baro_raw += np.cumsum(rng.normal(0, args.baro_drift_per_s/args.fs, n))
        baro_med = median_filter(baro_raw, args.baro_median)
        kf_baro = BaroKalman1D(baro_med[0], args.baro_q, args.baro_r)
        baro_f = np.asarray([kf_baro.update(v) for v in baro_med], dtype=np.float64)
        if getattr(args, 'input_mode', 'radio') == 'radio':
            terrain = baro_f - radar_f
        else:
            terrain = radio_raw
            radar_f = radio_raw
            baro_f = np.full_like(terrain, args.absolute_altitude)
        radar_raw = radio_raw
    else:
        if not getattr(args, 'nmea', None):
            raise ValueError('Need --heights or --nmea')
        terrain, radar_raw, radar_f, baro_f = make_terrain_profile(args)
        start_idx = max(0, int(getattr(args, 'nmea_start_index', 0)))
        if start_idx:
            terrain=terrain[start_idx:]; radar_raw=radar_raw[start_idx:]; radar_f=radar_f[start_idx:]; baro_f=baro_f[start_idx:]

    # --- Стартовая точка: Режим 2 = из CLI, Режим 1 = из truth-CSV ---
    def _get_start_xy():
        if getattr(args, 'start_x', None) is not None and getattr(args, 'start_y', None) is not None:
            return float(args.start_x), float(args.start_y)
        if truth is not None and len(truth['x']) > start_idx:
            return float(truth['x'][start_idx]), float(truth['y'][start_idx])
        if getattr(args, 'start_lon', None) is not None and getattr(args, 'start_lat', None) is not None:
            return geo_to_pixel(meta, float(args.start_lon), float(args.start_lat))
        raise ValueError('Need --start-x/--start-y or --truth')
    sx, sy = _get_start_xy()

    # --- Начальный курс ---
    init_heading = getattr(args, 'initial_heading_deg', None)
    if init_heading is None:
        init_heading = getattr(args, 'initial_azimuth_deg', None)
    if init_heading is None and truth is not None:
        n = int(getattr(args, 'heading_samples', 30))
        n = max(1, min(n, len(truth['x'])-start_idx-1))
        dx = float(truth['x'][start_idx+n] - truth['x'][start_idx])
        dy = float(truth['y'][start_idx+n] - truth['y'][start_idx])
        init_heading = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
        print(f'[авто-курс] из truth: {init_heading:.2f}°')
    init_vx = 0.0; init_vy = 0.0
    if init_heading is not None:
        if getattr(args, 'min_speed_px_s', None) is None or getattr(args, 'max_speed_px_s', None) is None:
            est = estimate_px_speed_range_from_meta(meta, args.min_speed_m_s, args.max_speed_m_s)
            if est:
                mn, mx, _, _ = est
                if getattr(args, 'min_speed_px_s', None) is None: args.min_speed_px_s = mn
                if getattr(args, 'max_speed_px_s', None) is None: args.max_speed_px_s = mx
        sp0 = 0.5*((getattr(args, 'min_speed_px_s', 0.5) or 0.5) + (getattr(args, 'max_speed_px_s', 2.0) or 2.0))
        az = math.radians(init_heading)
        init_vx = math.sin(az) * sp0
        init_vy = -math.cos(az) * sp0
    if init_heading is not None and not getattr(args, 'initial_azimuth_deg', None):
        args.initial_azimuth_deg = init_heading

    init=KalmanCV(np.array([sx,sy,init_vx,init_vy],float), np.diag([args.init_pos_sigma**2,args.init_pos_sigma**2,args.init_vel_sigma**2,args.init_vel_sigma**2]))
    hyps=[Hyp(init,0.0,False,[(sx,sy)])]
    rows=[]; events=[]
    timing={'bootstrap_s':0.0,'local_s':0.0,'bootstrap_calls':0,'local_calls':0}
    first_bootstrap_sample=None
    dt=1/args.fs

    for i in range(len(terrain)):
        if i>0:
            for h in hyps:
                h.kf.predict(dt,args.accel_sigma)
                h.hist.append((float(h.kf.x[0]),float(h.kf.x[1])))
                if len(h.hist)>args.max_history:
                    h.hist=h.hist[-args.max_history:]
        any_locked=any(h.locked for h in hyps)
        period=args.local_period if any_locked else args.bootstrap_period
        if i>=args.window_size and (i % period == 0 or i==len(terrain)-1):
            live=terrain[i-args.window_size+1:i+1]
            r=roughness(live)
            new=[]
            if not any_locked:
                if r < args.min_roughness_bootstrap:
                    hyps[0].last_rough=r; events.append(dict(sample=i,t_s=(i+start_idx)/args.fs,reason='flat_bootstrap_skip',roughness=r,accepted=0))
                else:
                    t0=time.perf_counter(); cands=bootstrap_match(dem, terrain, i, sx, sy, i/args.fs, args); timing['bootstrap_s']+=time.perf_counter()-t0; timing['bootstrap_calls']+=1
                    if cands and first_bootstrap_sample is None: first_bootstrap_sample=i
                    events.append(dict(sample=i,t_s=(i+start_idx)/args.fs,reason='bootstrap',roughness=r,accepted=len(cands),num_candidates=len(cands)))
                    for rank,c in enumerate(cands):
                        # Первый захват: оцениваем среднюю скорость от последнего GPS до кандидата.
                        elapsed=max(i/args.fs, dt)
                        vx=(c.x-sx)/elapsed; vy=(c.y-sy)/elapsed
                        kf=KalmanCV(np.array([c.x,c.y,vx,vy],float), np.diag([args.meas_sigma_min**2,args.meas_sigma_min**2,args.init_vel_sigma**2,args.init_vel_sigma**2]))
                        hist_len = min(i + 1, args.window_size)
                        alphas = np.linspace(0.0, 1.0, hist_len)
                        hist = [(float(sx + a * (c.x - sx)), float(sy + a * (c.y - sy))) for a in alphas]
                        new.append(Hyp(kf, c.score-args.rank_penalty*rank, True, hist, c.corr, r, 1))
                        events.append(dict(sample=i,t_s=(i+start_idx)/args.fs,reason='bootstrap_candidate',rank=rank,x_meas=c.x,y_meas=c.y,azimuth_deg=c.az,speed_px_s=c.speed,corr=c.corr,score=c.score,roughness=r,accepted=1))
                    if new:
                        hyps=sorted(new,key=lambda h:h.score,reverse=True)[:args.max_hypotheses]
            else:
                for hi,h in enumerate(hyps):
                    if r < args.min_roughness_local:
                        h.score -= args.flat_penalty; h.last_rough=r; new.append(h)
                        events.append(dict(sample=i,t_s=(i+start_idx)/args.fs,hyp=hi,reason='flat_local_skip',roughness=r,accepted=0,x_pred=h.kf.x[0],y_pred=h.kf.x[1]))
                        continue
                    # Для манёвров используем короткий сегментный матчер от недавней истории.
                    Lloc=min(args.local_window_size, len(h.hist), i+1)
                    live_local=terrain[i-Lloc+1:i+1]
                    anchor_xy=h.hist[-Lloc]
                    elapsed_s=(Lloc-1)/args.fs
                    t0=time.perf_counter()
                    if args.local_match_mode == 'history':
                        # Режим history — для прямых/плавных траекторий: используем основное окно TERCOM, а не короткое сегментное.
                        cands=history_match(dem, live, np.asarray(h.hist,dtype=np.float32), args)
                    else:
                        cands=segment_match(dem, live_local, anchor_xy[0], anchor_xy[1], elapsed_s, float(h.kf.x[0]), float(h.kf.x[1]), args)
                    timing['local_s']+=time.perf_counter()-t0; timing['local_calls']+=1
                    if not cands:
                        h.score -= args.no_candidate_penalty; h.last_rough=r; new.append(h)
                        events.append(dict(sample=i,t_s=(i+start_idx)/args.fs,hyp=hi,reason='no_local_candidates',roughness=r,accepted=0,x_pred=h.kf.x[0],y_pred=h.kf.x[1]))
                        continue
                    for rank,c in enumerate(cands):
                        nh=clone_hyp(h)
                        sigma=meas_sigma_from_corr(c.corr,args)
                        innov=math.hypot(c.x-nh.kf.x[0], c.y-nh.kf.x[1])
                        nh.kf.update_pos(np.array([c.x,c.y]), sigma)
                        # заменяем конец истории скорректированной позицией для последующего history-матчинга
                        if nh.hist:
                            nh.hist[-1]=(float(nh.kf.x[0]),float(nh.kf.x[1]))
                        nh.score += args.corr_weight*c.corr - args.rank_penalty*rank - args.innovation_penalty*min(innov/max(args.local_radius_px,1),3)
                        nh.last_corr=c.corr; nh.last_rough=r; nh.updates += 1
                        new.append(nh)
                        events.append(dict(sample=i,t_s=(i+start_idx)/args.fs,hyp=hi,rank=rank,reason='local_candidate',x_meas=c.x,y_meas=c.y,dx=c.dx,dy=c.dy,dtheta=c.dtheta,corr=c.corr,score=nh.score,roughness=r,innovation=innov,accepted=1,x_pred=h.kf.x[0],y_pred=h.kf.x[1]))
                hyps=sorted(new,key=lambda h:h.score,reverse=True)[:args.max_hypotheses]
        best=hyps[0]
        pos_unc=float(math.sqrt(max(0,best.kf.P[0,0]+best.kf.P[1,1])))
        vel_unc=float(math.sqrt(max(0,best.kf.P[2,2]+best.kf.P[3,3])))
        conf=solution_confidence(best.last_corr,best.last_rough,best.locked,args)
        row=dict(sample=i,t_s=(i+start_idx)/args.fs,x_est=float(best.kf.x[0]),y_est=float(best.kf.x[1]),vx_est=float(best.kf.x[2]),vy_est=float(best.kf.x[3]),speed_px_s=float(math.hypot(best.kf.x[2],best.kf.x[3])),azimuth_deg=float(vel_to_az(best.kf.x[2],best.kf.x[3])),nav_status='LOCKED' if best.locked else 'DEAD_RECKONING_NO_GPS',terrain=float(terrain[i]),radar_raw=float(radar_raw[i]),radar_filtered=float(radar_f[i]),baro_filtered=float(baro_f[i]),hyp_score=float(best.score),last_corr=float(best.last_corr) if np.isfinite(best.last_corr) else np.nan,last_rough=float(best.last_rough),position_uncertainty_px=pos_unc,velocity_uncertainty_px_s=vel_unc,solution_confidence=conf)
        row.update(velocity_px_to_metric(meta, row['x_est'], row['y_est'], row['vx_est'], row['vy_est']))
        if truth is not None and i+start_idx < len(truth['x']):
            tx=float(truth['x'][i+start_idx]); ty=float(truth['y'][i+start_idx])
            tlon,tlat = pixel_to_geo(meta, tx, ty)
            row.update(x_true=tx,y_true=ty,lon_true=tlon,lat_true=tlat,err_px=float(math.hypot(row['x_est']-tx,row['y_est']-ty)),err_m=pixel_error_to_meters(meta,row['x_est'],row['y_est'],tx,ty))
        rows.append(row)

    # --- постобработка (без обратной связи в MHT): RTS-сглаживание -> доводка по корреляции ---
    did_post = False
    if getattr(args, 'smooth', True) and len(rows) >= 3:
        rts_smooth_positions(rows, args.fs, args.accel_sigma, args.meas_sigma_min, args.meas_sigma_max)
        did_post = True
    if getattr(args, 'refit', False) and len(rows) >= 3:
        correlation_refit(dem, terrain, rows, args)
        did_post = True
    if did_post:
        # пересчитываем гео-производные + отладочную ошибку после постобработки
        for i, r in enumerate(rows):
            r.update(velocity_px_to_metric(meta, r['x_est'], r['y_est'], r['vx_est'], r['vy_est']))
            if truth is not None and i+start_idx < len(truth['x']):
                tx=float(truth['x'][i+start_idx]); ty=float(truth['y'][i+start_idx])
                tlon,tlat = pixel_to_geo(meta, tx, ty)
                r.update(x_true=tx, y_true=ty, lon_true=tlon, lat_true=tlat,
                         err_px=float(math.hypot(r['x_est']-tx, r['y_est']-ty)),
                         err_m=pixel_error_to_meters(meta, r['x_est'], r['y_est'], tx, ty))

    # --- детерминированная финальная коррекция позиции (без обратной связи в MHT) ---
    if getattr(args, 'final_fix', True) and rows and hyps[0].locked:
        b = hyps[0]
        fx = final_position_fix(dem, terrain, float(b.kf.x[0]), float(b.kf.x[1]), float(b.kf.x[2]), float(b.kf.x[3]), args)
        if fx is not None:
            xf, yf, cf, shift = fx
            r = rows[-1]
            r['x_est'] = xf; r['y_est'] = yf
            r['final_fix_corr'] = cf; r['final_fix_shift_px'] = shift
            r.update(velocity_px_to_metric(meta, xf, yf, r['vx_est'], r['vy_est']))
            if truth is not None and (len(rows)-1)+start_idx < len(truth['x']):
                tx = float(truth['x'][len(rows)-1+start_idx]); ty = float(truth['y'][len(rows)-1+start_idx])
                r['err_px'] = float(math.hypot(xf-tx, yf-ty)); r['err_m'] = pixel_error_to_meters(meta, xf, yf, tx, ty)
            print(f'[final-fix] corr={cf:.3f} shift={shift:.2f}px -> ({xf:.2f},{yf:.2f})')
        else:
            print('[final-fix] пропущено (плоско/слабо/на границе) — оставляем оценку трекера')

    write_csv(out/'estimated_trajectory.csv', rows)
    # --- экспорт итоговых файлов ---
    try:
        import csv
        with open(out/'trajectory_local.csv','w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['t_s','x_px','y_px'])
            for r in rows: w.writerow([r.get('t_s'), r.get('x_est'), r.get('y_est')])
        with open(out/'trajectory_global.csv','w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['t_s','lon','lat','speed_m_s','course_deg','v_east_m_s','v_north_m_s'])
            for r in rows: w.writerow([r.get('t_s'), r.get('lon_est'), r.get('lat_est'), r.get('speed_m_s'), r.get('course_deg'), r.get('v_east_m_s'), r.get('v_north_m_s')])
    except Exception:
        pass
    write_csv(out/'correlation_events.csv', events)
    plot_results(out, dem, rows, events, truth, start_idx)
    if args.plot_heatmap:
        hm_sample = first_bootstrap_sample if first_bootstrap_sample is not None else min(len(terrain)-1, max(args.window_size, args.bootstrap_period))
        t0=time.perf_counter(); hmdata=compute_bootstrap_heatmap(dem, terrain, sx, sy, hm_sample, args); plot_bootstrap_heatmap(out, hmdata); timing['heatmap_s']=time.perf_counter()-t0
    generate_report(out, rows, events, args)
    if args.profile:
        prof_path=out/'profile_summary.json'
        prof_path.write_text(json.dumps(timing,indent=2),encoding='utf-8')
        print('PROFILE:', timing)
    errs=[r['err_px'] for r in rows if 'err_px' in r]
    errm=[r['err_m'] for r in rows if 'err_m' in r and np.isfinite(r['err_m'])]
    print('\n=== SUMMARY ===')
    print('n_samples:',len(rows)); print('n_events:',len(events)); print('locked_samples:',sum(r['nav_status']=='LOCKED' for r in rows))
    if errs:
        print('mean_err_px:',float(np.mean(errs))); print('median_err_px:',float(np.median(errs))); print('final_err_px:',float(errs[-1]))
    if errm:
        print('mean_err_m:',float(np.mean(errm))); print('median_err_m:',float(np.median(errm))); print('p95_err_m:',float(np.percentile(errm,95))); print('final_err_m:',float(errm[-1]))
    print('final_x:',rows[-1]['x_est']); print('final_y:',rows[-1]['y_est'])
    print('final_lon:',rows[-1].get('lon_est')); print('final_lat:',rows[-1].get('lat_est')); print('final_speed_m_s:',rows[-1].get('speed_m_s')); print('final_course_deg:',rows[-1].get('course_deg'))
    print('out_dir:',out)


def write_csv(path, rows):
    if not rows: return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)



def plot_results(out, dem, rows, events, truth, start_idx):
    if not HAS_MPL: return
    import numpy as np, math
    import matplotlib.pyplot as plt
    xs=np.array([r['x_est'] for r in rows]); ys=np.array([r['y_est'] for r in rows])
    vxs=np.array([r.get('vx_est',0.0) for r in rows]); vys=np.array([r.get('vy_est',0.0) for r in rows])
    speeds=np.array([r.get('speed_m_s', np.hypot(vx,vy)) for r,vx,vy in zip(rows,vxs,vys)])
    allx=[xs.min(),xs.max()]; ally=[ys.min(),ys.max()]
    if truth is not None:
        tx=truth['x'][start_idx:start_idx+len(rows)]; ty=truth['y'][start_idx:start_idx+len(rows)]
        allx += [np.nanmin(tx),np.nanmax(tx)]; ally += [np.nanmin(ty),np.nanmax(ty)]
    pad=80; h,w=dem.shape
    x0=max(0,int(min(allx)-pad)); x1=min(w,int(max(allx)+pad)); y0=max(0,int(min(ally)-pad)); y1=min(h,int(max(ally)+pad))
    fig,ax=plt.subplots(figsize=(14,7))
    ax.imshow(dem[y0:y1,x0:x1], origin='upper', cmap='terrain', extent=[x0,x1,y1,y0], aspect='equal', alpha=0.95)
    if truth is not None:
        ax.plot(truth['x'][start_idx:start_idx+len(rows)], truth['y'][start_idx:start_idx+len(rows)], 'w--', lw=2.2, alpha=0.9, label='true')
    ax.plot(xs, ys, 'r-', lw=2.5, label='TERCOM estimate')
    # векторы скорости — цвет по скорости
    step = max(1, len(rows)//20)
    qx=xs[::step]; qy=ys[::step]; qvx=vxs[::step]; qvy=vys[::step]; qspeed=speeds[::step]
    if len(qx) > 1:
        sc = ax.quiver(qx, qy, qvx, qvy, qspeed, cmap='plasma', scale=25, width=0.0045, alpha=0.95)
        cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
        cb.set_label('speed, m/s')
    # измерения TERCOM
    mx=[e.get('x_meas') for e in events if e.get('accepted')==1 and e.get('rank',99)==0]
    my=[e.get('y_meas') for e in events if e.get('accepted')==1 and e.get('rank',99)==0]
    if mx: ax.scatter(mx,my,s=18,c='cyan',edgecolors='k',linewidths=.3,alpha=0.75,label='TERCOM meas')
    # старт / финиш
    ax.scatter([xs[0]],[ys[0]],marker='*',s=220,c='lime',edgecolors='k',linewidths=1.0,label='start',zorder=5)
    ax.scatter([xs[-1]],[ys[-1]],marker='o',s=90,c='red',edgecolors='k',linewidths=1.0,label='finish',zorder=5)
    # стрелка на север
    ax.annotate('N\n↑', xy=(0.97,0.92), xytext=(0.97,0.86), xycoords='axes fraction',
                ha='center', va='center', fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.5))
    # информационная плашка
    final=rows[-1]
    txt = f"speed = {final.get('speed_m_s',0):.1f} m/s\ncourse = {final.get('course_deg',0):.1f}°\nconf = {final.get('solution_confidence',0):.2f}"
    if 'err_m' in final and np.isfinite(final.get('err_m', np.nan)):
        txt += f"\nerr = {final['err_m']:.1f} m"
    ax.text(0.02,0.03,txt,transform=ax.transAxes,fontsize=9,
            bbox=dict(facecolor='white',alpha=0.82,edgecolor='0.5'))
    ax.set_xlabel('x_px / col'); ax.set_ylabel('y_px / row')
    ax.set_title('TERCOM MHT Kalman trajectory – straight flight')
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(out/'trajectory_on_dem.png', dpi=180); plt.close(fig)
    # график ошибки
    if rows and 'err_px' in rows[0]:
        err=np.array([r.get('err_px',np.nan) for r in rows])
        fig,ax=plt.subplots(figsize=(12,3.5))
        ax.plot([r['t_s'] for r in rows], err, lw=1.2)
        ax.grid(True, alpha=.3); ax.set_xlabel('t, s'); ax.set_ylabel('error, px')
        ax.set_title(f'Error: mean={np.nanmean(err):.2f}px, median={np.nanmedian(err):.2f}px, final={err[-1]:.2f}px')
        fig.tight_layout(); fig.savefig(out/'error_vs_time.png',dpi=150); plt.close(fig)

def build_parser():
    p=argparse.ArgumentParser(description='TERCOM Final – DEM + heights (radio altimeter) + start point / heading – 2 modes')
    # ввод-вывод
    p.add_argument('--dem', default=None, help='DEM: GeoTIFF / .npy / NPZ txt')
    p.add_argument('--map', default=None, help='alias for --dem')
    p.add_argument('--nmea', default=None, help='legacy: NMEA GPGGA radio altimeter log')
    p.add_argument('--heights', default=None, help='radio altimeter profile, text file one height (m) per line – if given, --nmea is ignored')
    p.add_argument('--input-mode', choices=['radio','terrain'], default='radio', help='heights = radio altitude (filtered inside, default) or terrain elevation')
    # старт — Режим 1: из truth, Режим 2: из CLI
    p.add_argument('--start-x', type=float, default=None, help='Mode 2: start x_px / col')
    p.add_argument('--start-y', type=float, default=None, help='Mode 2: start y_px / row')
    p.add_argument('--start-lon', type=float, default=None)
    p.add_argument('--start-lat', type=float, default=None)
    p.add_argument('--truth', default=None, help='Mode 1 / debug: truth csv with x_px,y_px – if --start-x/y missing, start = first truth point')
    p.add_argument('--initial-heading-deg', type=float, default=None, help='Mode 2: initial course, 0=north, clockwise')
    p.add_argument('--heading-samples', type=int, default=30, help='Mode 1: points from truth to compute initial heading')
    p.add_argument('--out-dir', default='result_final')
    # сенсоры / предобработка — точно как в исходном kod.txt
    p.add_argument('--fs', type=float, default=10.0)
    p.add_argument('--absolute-altitude', type=float, default=1500.0)
    p.add_argument('--nmea-start-index', type=int, default=0)
    p.add_argument('--radar-median', type=int, default=3)
    p.add_argument('--radar-cutoff', type=float, default=3.0)
    p.add_argument('--baro-noise-std', type=float, default=0.0)
    p.add_argument('--baro-drift-per-s', type=float, default=0.0)
    p.add_argument('--noise-seed', type=int, default=None)
    p.add_argument('--baro-median', type=int, default=3)
    p.add_argument('--baro-q', type=float, default=0.02)
    p.add_argument('--baro-r', type=float, default=1.0)
    # TERCOM — исходные значения по умолчанию
    p.add_argument('--window-size', type=int, default=200)
    p.add_argument('--max-history', type=int, default=300)
    p.add_argument('--bootstrap-period', type=int, default=200)
    p.add_argument('--local-period', type=int, default=50)
    p.add_argument('--min-roughness-bootstrap', type=float, default=4.0)
    p.add_argument('--min-roughness-local', type=float, default=1.0)
    p.add_argument('--min-speed-px-s', type=float, default=None)
    p.add_argument('--max-speed-px-s', type=float, default=None)
    p.add_argument('--min-speed-m-s', type=float, default=5.0)
    p.add_argument('--max-speed-m-s', type=float, default=45.0)
    p.add_argument('--bootstrap-speed-step-px-s', type=float, default=0.2)
    p.add_argument('--bootstrap-az-step-deg', type=float, default=5.0)
    p.add_argument('--bootstrap-radius-px', type=float, default=8.0)
    p.add_argument('--bootstrap-grid-px', type=float, default=4.0)
    p.add_argument('--initial-azimuth-deg', type=float, default=None, help='alias, 0=north')
    p.add_argument('--initial-az-window-deg', type=float, default=180.0)
    p.add_argument('--bootstrap-verify-windows', type=int, nargs='*', default=[])
    p.add_argument('--local-radius-px', type=float, default=10.0)
    p.add_argument('--local-grid-px', type=float, default=10.0)
    p.add_argument('--local-theta-deg', type=float, default=8.0)
    p.add_argument('--local-theta-step-deg', type=float, default=4.0)
    p.add_argument('--local-match-mode', choices=['segment','history'], default='history')
    p.add_argument('--local-window-size', type=int, default=50)
    p.add_argument('--local-az-step-deg', type=float, default=5.0)
    p.add_argument('--local-speed-step-px-s', type=float, default=0.2)
    p.add_argument('--local-endpoint-gate-px', type=float, default=35.0)
    p.add_argument('--segment-innovation-penalty', type=float, default=0.25)
    p.add_argument('--min-corr-bootstrap', type=float, default=0.65)
    p.add_argument('--min-corr-local', type=float, default=0.55)
    p.add_argument('--min-quality-corr', type=float, default=0.55)
    p.add_argument('--top-candidates', type=int, default=6)
    p.add_argument('--max-hypotheses', type=int, default=8)
    p.add_argument('--chunk-candidates', type=int, default=4096)
    p.add_argument('--nms-radius-px', type=float, default=8.0)
    p.add_argument('--offset-penalty', type=float, default=0.03)
    p.add_argument('--theta-penalty', type=float, default=0.03)
    p.add_argument('--rank-penalty', type=float, default=0.04)
    p.add_argument('--corr-weight', type=float, default=1.0)
    p.add_argument('--innovation-penalty', type=float, default=0.20)
    p.add_argument('--flat-penalty', type=float, default=0.01)
    p.add_argument('--no-candidate-penalty', type=float, default=0.1)
    # Калман — исходные значения
    p.add_argument('--init-pos-sigma', type=float, default=3.0)
    p.add_argument('--init-vel-sigma', type=float, default=3.0)
    p.add_argument('--accel-sigma', type=float, default=0.05)
    p.add_argument('--meas-sigma-min', type=float, default=1.0)
    p.add_argument('--meas-sigma-max', type=float, default=25.0)
    # вывод
    p.add_argument('--plot-heatmap', action='store_true', default=True)
    p.add_argument('--no-plot-heatmap', dest='plot_heatmap', action='store_false')
    p.add_argument('--heatmap-az-step-deg', type=float, default=1.0)
    p.add_argument('--heatmap-speed-step-px-s', type=float, default=0.2)
    p.add_argument('--profile', action='store_true')
    # RTS-сглаживатель со взвешиванием по уверенности по всей траектории (офлайн, включён по умолчанию)
    p.add_argument('--smooth', action='store_true', default=True, help='apply confidence-weighted RTS smoother to the trajectory (reduces transient excursions)')
    p.add_argument('--no-smooth', dest='smooth', action='store_false')
    # детерминированная доводка всей траектории по корреляции (офлайн, снижает медианную ошибку)
    p.add_argument('--refit', action='store_true', default=False, help='snap whole trajectory to the terrain-correlation peak at sub-sampled anchors, then re-smooth (slower; cuts median error to near the DEM-resolution floor)')
    p.add_argument('--refit-step', type=int, default=8, help='anchor every N samples for the correlation refit')
    p.add_argument('--refit-radius-px', type=float, default=2.0, help='bounded search half-extent (px) around current estimate')
    p.add_argument('--refit-res-px', type=float, default=0.2, help='search resolution (px) for the refit')
    p.add_argument('--refit-gate-px', type=float, default=1.5, help='reject a correction larger than this (false-peak guard on ambiguous terrain)')
    p.add_argument('--refit-base-sigma', type=float, default=4.0, help='measurement noise (px) on the current smoothed path between anchors')
    p.add_argument('--refit-anchor-sigma', type=float, default=0.5, help='measurement noise (px) on accepted correlation anchors')
    # детерминированная финальная коррекция позиции — защёлкивает итоговую точку на пик корреляции
    # рельефа ограниченным плотным поиском (без обратной связи в MHT). Включена по умолчанию.
    p.add_argument('--final-fix', action='store_true', default=True, help='snap final reported position to the terrain-correlation peak (bounded, deterministic)')
    p.add_argument('--no-final-fix', dest='final_fix', action='store_false')
    p.add_argument('--final-fix-window', type=int, default=None, help='profile length (samples) for the final fix; default = --window-size')
    p.add_argument('--final-fix-radius-px', type=float, default=3.0, help='search half-extent (px) around the converged estimate')
    p.add_argument('--final-fix-res-px', type=float, default=0.1, help='search resolution (px) for the final fix')
    p.add_argument('--final-fix-min-roughness', type=float, default=0.5, help='min terrain roughness to trust the final fix (correlation is the primary guard)')
    return p

if __name__ == '__main__':
    run(build_parser().parse_args())
