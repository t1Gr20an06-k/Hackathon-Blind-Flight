// Полёт вслепую — навигация по рельефу.
// Пункты 1-2: парсинг данных радиовысотомера (NMEA-0183) и вычисление высоты рельефа.
//
// Идея в одной строке:
//     высота рельефа над морем = высота полёта над морем (барометр) − расстояние до земли (радиовысотомер)
//
// Радиовысотомер отдаёт "расстояние до земли" (AGL) в поле высоты сообщения GPGGA.
// Барометр даёт абсолютную высоту полёта над уровнем моря (AMSL) — по условию 1500 м.
// Разность даёт абсолютную высоту самого рельефа — это и есть профиль для сравнения с картой.
//
// Сборка:  g++ -std=c++17 -O2 terrain_profile.cpp -o terrain_profile

#include <string>
#include <vector>
#include <optional>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cctype>

// ──────────────────────────────────────────────────────────────
//  Вспомогательные функции
// ──────────────────────────────────────────────────────────────
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Двузначное hex-число в верхнем регистре: 71 -> "47", 9 -> "09".
static std::string to_hex2(int v) {
    std::ostringstream o;
    o << std::uppercase << std::hex << std::setw(2) << std::setfill('0') << (v & 0xFF);
    return o.str();
}

// Режет строку по разделителю, СОХРАНЯЯ пустые поля (в NMEA важны ",,,,").
static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(delim, start);
        if (pos == std::string::npos) { out.push_back(s.substr(start)); break; }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

// ──────────────────────────────────────────────────────────────
//  Контрольная сумма NMEA
// ──────────────────────────────────────────────────────────────

// Добавляет контрольную сумму к строке вида "$GPGGA,..." (без *xx).
// Пригодится, когда будешь сам генерировать данные радиовысотомера.
std::string append_checksum(const std::string& bare) {
    int cs = 0;
    for (size_t i = 1; i < bare.size(); ++i)   // XOR всех символов после '$'
        cs ^= static_cast<unsigned char>(bare[i]);
    return bare + "*" + to_hex2(cs);
}

// Контрольная сумма NMEA = XOR всех символов между '$' и '*'.
// В "$GPGGA,...*47" число 47 (hex) должно совпасть с посчитанным.
// Защищает от битых строк (помехи в канале радиовысотомера).
bool is_checksum_valid(const std::string& sentence_in) {
    std::string s = trim(sentence_in);
    if (s.empty() || s[0] != '$') return false;
    size_t star = s.find('*');
    if (star == std::string::npos) return false;

    int cs = 0;
    for (size_t i = 1; i < star; ++i)          // всё между $ и *
        cs ^= static_cast<unsigned char>(s[i]);

    std::string given = s.substr(star + 1, 2); // два hex-символа после *
    for (char& c : given) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return to_hex2(cs) == given;
}

// ──────────────────────────────────────────────────────────────
//  Пункт 1. Парсинг сообщения GPGGA
// ──────────────────────────────────────────────────────────────
struct AltReading {
    double t;     // время в секундах от полуночи (нужно потом для скорости)
    double agl;   // расстояние до земли, метры (Above Ground Level)
};

// "123519.111" → 12*3600 + 35*60 + 19.111  (секунды от полуночи).
static double parse_time(const std::string& hhmmss) {
    int hh = std::stoi(hhmmss.substr(0, 2));
    int mm = std::stoi(hhmmss.substr(2, 2));
    double ss = std::stod(hhmmss.substr(4));
    return hh * 3600 + mm * 60 + ss;
}

// Разбирает строку:  $GPGGA,123519.111,,,,,,,,545.4,M,46.9,M,,*47
// Поля (через запятую):
//     [0]  $GPGGA       — тип сообщения
//     [1]  123519.111   — время UTC (ччммсс.ссс)
//     [9]  545.4        — ВЫСОТА = показание радиовысотомера (м до земли)
//     [10] M            — единица измерения (метры)
// verify_checksum=true  — строгий режим: отбрасывает строки с неверной суммой.
// verify_checksum=false — мягкий режим: парсит даже с «битой» суммой
//                         (пример из условия с *47 — как раз такой случай).
// Возвращает значение либо std::nullopt, если строка повреждена.
std::optional<AltReading> parse_gpgga(const std::string& sentence, bool verify_checksum = true) {
    if (verify_checksum && !is_checksum_valid(sentence))
        return std::nullopt;

    std::string body = trim(sentence);
    body = body.substr(0, body.find('*'));     // отрезаем контрольную сумму
    std::vector<std::string> f = split(body, ',');

    if (f.empty() || f[0] != "$GPGGA" || f.size() < 11)  // нужно хотя бы до поля высоты
        return std::nullopt;

    try {
        double t = parse_time(f[1]);           // время → секунды
        double agl = std::stod(f[9]);          // поле 9 — высота над землёй
        return AltReading{t, agl};
    } catch (...) {                            // нечисловые/пустые поля
        return std::nullopt;
    }
}

// ──────────────────────────────────────────────────────────────
//  Пункт 2. Высота рельефа над уровнем моря
// ──────────────────────────────────────────────────────────────

// высота рельефа = высота полёта над морем − расстояние до земли.
// Пример: 1500 − 545.4 = 954.6 м над уровнем моря.
// Результат на той же системе высот, что и карта (SRTM / Copernicus), —
// над уровнем моря, поэтому профиль можно напрямую сравнивать с DEM.
double terrain_elevation(double agl, double baro_amsl = 1500.0) {
    return baro_amsl - agl;
}

// ──────────────────────────────────────────────────────────────
//  Сборка профиля рельефа из потока NMEA
// ──────────────────────────────────────────────────────────────
struct TerrainPoint {
    double t;          // время, с
    double agl;        // расстояние до земли, м
    double elevation;  // высота рельефа над морем, м
};

// Поток NMEA-строк → профиль рельефа вдоль трассы.
// Битые строки молча пропускаются.
// Возвращённый список высот — тот самый "отпечаток местности" для корреляции с картой.
std::vector<TerrainPoint> build_terrain_profile(const std::vector<std::string>& lines,
                                                double baro_amsl = 1500.0,
                                                bool verify_checksum = true) {
    std::vector<TerrainPoint> profile;
    for (const std::string& line : lines) {
        std::optional<AltReading> r = parse_gpgga(line, verify_checksum);
        if (!r) continue;                                      // пропускаем повреждённые
        double elev = terrain_elevation(r->agl, baro_amsl);
        profile.push_back(TerrainPoint{r->t, r->agl, elev});
    }
    return profile;
}

// ──────────────────────────────────────────────────────────────
//  Демонстрация
// ──────────────────────────────────────────────────────────────

// Ширина строки в видимых символах (UTF-8: кириллица = 1 символ, а не 2 байта).
static size_t display_width(const std::string& s) {
    size_t w = 0;
    for (unsigned char c : s)
        if ((c & 0xC0) != 0x80) ++w;          // считаем только ведущие байты
    return w;
}
// Выравнивание по правому краю с учётом кириллицы.
static std::string pad_left(const std::string& s, size_t width) {
    size_t w = display_width(s);
    return (w >= width) ? s : std::string(width - w, ' ') + s;
}

int main() {
    // Пример потока радиовысотомера. Поле высоты (9) = расстояние до земли.
    // Контрольные суммы добавляем автоматически, чтобы строки были валидными.
    std::vector<std::string> bare = {
        "$GPGGA,123519.111,,,,,,,,545.4,M,46.9,M,,",
        "$GPGGA,123519.211,,,,,,,,548.1,M,46.9,M,,",
        "$GPGGA,123519.311,,,,,,,,552.0,M,46.9,M,,",
        "$GPGGA,123519.411,,,,,,,,539.7,M,46.9,M,,",
        "$GPGGA,GARBAGE_BROKEN_LINE",                  // пример битой строки
    };
    std::vector<std::string> sample;
    for (const std::string& b : bare)
        sample.push_back((!b.empty() && b.back() == ',') ? append_checksum(b) : b);

    const double BARO = 1500.0;   // абсолютная высота полёта над морем, м (по условию)

    std::vector<TerrainPoint> profile = build_terrain_profile(sample, BARO);

    std::cout << "Распознано точек: " << profile.size()
              << " из " << sample.size() << " строк\n\n";

    std::cout << pad_left("t, с", 12) << pad_left("до земли, м", 16)
              << pad_left("рельеф, м", 14) << "\n";

    std::cout << std::fixed;
    for (const TerrainPoint& p : profile) {
        std::cout << std::setprecision(3) << std::setw(12) << p.t
                  << std::setprecision(1) << std::setw(16) << p.agl
                  << std::setw(14) << p.elevation << "\n";
    }
    return 0;
}
