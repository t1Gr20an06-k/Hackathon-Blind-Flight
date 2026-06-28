"""
Полёт вслепую — навигация по рельефу.
Пункты 1-2: парсинг данных радиовысотомера (NMEA-0183) и вычисление высоты рельефа.

Идея в одной строке:
    высота рельефа над морем = высота полёта над морем (барометр) − расстояние до земли (радиовысотомер)

Радиовысотомер отдаёт "расстояние до земли" (AGL) в поле высоты NMEA-сообщения GPGGA.
Барометр даёт абсолютную высоту полёта над уровнем моря (AMSL) — по условию 1500 м.
Разность даёт абсолютную высоту самого рельефа — это и есть профиль для сравнения с картой.
"""

from dataclasses import dataclass


#  Контрольная сумма NMEA

def append_checksum(bare: str) -> str:
    """
    Добавляет контрольную сумму к строке вида '$GPGGA,...' (без *xx).
    Пригодится, когда будешь сам генерировать данные радиовысотомера.
    """
    cs = 0
    for ch in bare[1:]:        # XOR всех символов после '$'
        cs ^= ord(ch)
    return f"{bare}*{cs:02X}"


def is_checksum_valid(sentence: str) -> bool:
    """
    Контрольная сумма NMEA = XOR всех символов между '$' и '*'.
    В "$GPGGA,...*47" число 47 (hex) должно совпасть с посчитанным.
    Защищает от битых строк (помехи в канале радиовысотомера).
    """
    sentence = sentence.strip()
    if not sentence.startswith('$') or '*' not in sentence:
        return False
    star = sentence.index('*')
    payload = sentence[1:star]             # всё между $ и *
    given = sentence[star + 1:star + 3]    # два hex-символа после *

    cs = 0
    for ch in payload:
        cs ^= ord(ch)
    return f"{cs:02X}" == given.upper()



#  Пункт 1. Парсинг сообщения GPGGA

@dataclass
class AltReading:
    """Одно показание радиовысотомера."""
    t: float        # время в секундах от полуночи (нужно потом для скорости)
    agl: float      # расстояние до земли, метры (Above Ground Level)


def parse_gpgga(sentence: str, verify_checksum: bool = True) -> AltReading | None:
    """
    Разбирает строку:  $GPGGA,123519.111,,,,,,,,545.4,M,46.9,M,,*47
    Поля (через запятую):
        [0]  $GPGGA       — тип сообщения
        [1]  123519.111   — время UTC (ччммсс.ссс)
        [9]  545.4        — ВЫСОТА = показание радиовысотомера (м до земли)
        [10] M            — единица измерения (метры)
    verify_checksum=True  — строгий режим: отбрасывает строки с неверной суммой.
    verify_checksum=False — мягкий режим: парсит даже с «битой» суммой
                            (пример из условия с *47 — как раз такой случай).
    Возвращает AltReading либо None, если строка повреждена.
    """
    if verify_checksum and not is_checksum_valid(sentence):
        return None

    body = sentence.strip().split('*')[0]   # отрезаем контрольную сумму
    f = body.split(',')                      # режем по запятым

    if f[0] != '$GPGGA' or len(f) < 11:      # нужно хотя бы до поля высоты
        return None

    try:
        t = _parse_time(f[1])                # время → секунды
        agl = float(f[9])                    # поле 9 — высота над землёй
    except (ValueError, IndexError):
        return None

    return AltReading(t=t, agl=agl)


def _parse_time(hhmmss: str) -> float:
    """ '123519.111' → 12*3600 + 35*60 + 19.111  (секунды от полуночи). """
    hh = int(hhmmss[0:2])
    mm = int(hhmmss[2:4])
    ss = float(hhmmss[4:])
    return hh * 3600 + mm * 60 + ss



#  Пункт 2. Высота рельефа над уровнем моря

def terrain_elevation(agl: float, baro_amsl: float = 1500.0) -> float:
    """
    высота рельефа = высота полёта над морем − расстояние до земли.
    Пример: 1500 − 545.4 = 954.6 м над уровнем моря.
    Результат на той же системе высот, что и карта (SRTM / Copernicus) — над уровнем моря,
    поэтому профиль можно напрямую сравнивать с DEM.
    """
    return baro_amsl - agl


# ──────────────────────────────────────────────────────────────
#  Сборка профиля рельефа из потока NMEA
# ──────────────────────────────────────────────────────────────
@dataclass
class TerrainPoint:
    t: float           # время, с
    agl: float         # расстояние до земли, м
    elevation: float   # высота рельефа над морем, м


def build_terrain_profile(nmea_lines, baro_amsl: float = 1500.0, verify_checksum: bool = True):
    """
    Поток NMEA-строк → профиль рельефа вдоль трассы.
    Битые строки молча пропускаются.
    Возвращённый список высот — тот самый "отпечаток местности" для корреляции с картой.
    """
    profile = []
    for line in nmea_lines:
        r = parse_gpgga(line, verify_checksum=verify_checksum)
        if r is None:
            continue                                   # пропускаем повреждённые
        elev = terrain_elevation(r.agl, baro_amsl)
        profile.append(TerrainPoint(t=r.t, agl=r.agl, elevation=elev))
    return profile


# ──────────────────────────────────────────────────────────────
#  Демонстрация
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Пример потока радиовысотомера. Поле высоты (9) = расстояние до земли.
    # Контрольные суммы добавляем автоматически, чтобы строки были валидными.
    bare = [
        "$GPGGA,123519.111,,,,,,,,545.4,M,46.9,M,,",
        "$GPGGA,123519.211,,,,,,,,548.1,M,46.9,M,,",
        "$GPGGA,123519.311,,,,,,,,552.0,M,46.9,M,,",
        "$GPGGA,123519.411,,,,,,,,539.7,M,46.9,M,,",
        "$GPGGA,GARBAGE_BROKEN_LINE",                  # пример битой строки
    ]
    sample = [append_checksum(b) if b.endswith(',') else b for b in bare]

    BARO = 1500.0   # абсолютная высота полёта над морем, м (по условию)

    profile = build_terrain_profile(sample, BARO)

    print(f"Распознано точек: {len(profile)} из {len(sample)} строк\n")
    print(f"{'t, с':>12} {'до земли, м':>14} {'рельеф, м':>12}")
    for p in profile:
        print(f"{p.t:>12.3f} {p.agl:>14.1f} {p.elevation:>12.1f}")
