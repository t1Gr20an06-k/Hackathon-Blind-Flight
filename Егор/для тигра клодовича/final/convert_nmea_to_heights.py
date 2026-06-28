#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_nmea_to_heights.py
Финальный чекпоинт – конвертер входных данных.

Вход : NMEA-0183 GPGGA лог радиовысотомера
Выход: heights_m.txt – по одному значению высоты на строку

По умолчанию – сырые зашумлённые отсчёты, БЕЗ фильтрации.
Фильтрация (median + LPF + барометр) выполняется уже в основном
навигационном скрипте tercom_final.py, как в оригинале.

Пример:
  python3 convert_nmea_to_heights.py \
    --nmea uploads/nmea_data_sibir.txt \
    --out final_input/heights_m.txt

Опционально можно включить предфильтрацию для отладки:
  --filter --radar-median 3 --radar-cutoff 3.0
"""
import argparse
from pathlib import Path
import numpy as np

def parse_gpgga_alt(line: str):
    if not line.startswith('$GPGGA'): return None
    parts = line.split('*',1)[0].split(',')
    try:
        return float(parts[9]) if len(parts)>9 and parts[9] else None
    except Exception:
        return None

def read_nmea(path: Path):
    vals = []
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        v = parse_gpgga_alt(line.strip())
        if v is not None:
            vals.append(v)
    if not vals:
        raise ValueError('No $GPGGA altitudes found')
    return np.asarray(vals, dtype=np.float64)

def median_filter(x, win):
    if win <= 1: return x.copy()
    if win % 2 == 0: win += 1
    r = win // 2
    xp = np.pad(x, (r,r), mode='edge')
    return np.array([np.median(xp[i:i+win]) for i in range(len(x))], dtype=float)

def lowpass(x, fs, cutoff):
    try:
        from scipy.signal import butter, filtfilt
        if cutoff <= 0 or cutoff >= 0.49*fs or len(x) < 20:
            return x.copy()
        b,a = butter(2, cutoff/(0.5*fs), btype='low')
        return filtfilt(b, a, x, padlen=min(len(x)-1, 12))
    except Exception:
        return x.copy()

def main():
    ap = argparse.ArgumentParser(description='NMEA GPGGA -> heights.txt for TERCOM (raw, noisy by default)')
    ap.add_argument('--nmea', required=True, help='input NMEA $GPGGA log (radio altimeter)')
    ap.add_argument('--out', default='heights_m.txt', help='output text, one height per line')
    ap.add_argument('--mode', choices=['radio','terrain'], default='radio',
                    help='radio = output radio altitude as-is (default, noisy – filtering is in tercom_final.py); terrain = H_baro - H_radio')
    ap.add_argument('--absolute-altitude', type=float, default=1500.0,
                    help='used only with --mode terrain')
    # optional pre-filtering, OFF by default – filtering belongs in tercom_final.py
    ap.add_argument('--filter', action='store_true', help='apply median+LPF to radio channel before writing (normally OFF – filter in main script)')
    ap.add_argument('--fs', type=float, default=10.0)
    ap.add_argument('--radar-median', type=int, default=3)
    ap.add_argument('--radar-cutoff', type=float, default=3.0)
    args = ap.parse_args()

    radio_raw = read_nmea(Path(args.nmea))

    if args.filter:
        radio_out = lowpass(median_filter(radio_raw, args.radar_median), args.fs, args.radar_cutoff)
    else:
        radio_out = radio_raw

    if args.mode == 'radio':
        out_vals = radio_out
    else:  # terrain
        out_vals = args.absolute_altitude - radio_out

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(f'{v:.3f}' for v in out_vals), encoding='utf-8')
    print(f'Wrote {len(out_vals)} samples -> {out_path}')
    print(f'range: {np.nanmin(out_vals):.2f} .. {np.nanmax(out_vals):.2f} m')
    print(f'mode: {args.mode}, filtered: {args.filter}')

if __name__ == '__main__':
    main()
