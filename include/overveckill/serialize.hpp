#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "overveckill/metadata.hpp"
#include "overveckill/metric.hpp"

namespace overveckill {

// Thrown when a file or stream can't be read back: wrong magic, unsupported
// format version, truncated data, or contents that fail validation.
struct SerializationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace io {

// On-disk format rules, shared by every saved object:
//   - all integers are fixed-width little-endian, whatever the host is
//   - floats/doubles are stored as their IEEE-754 bits, little-endian
//   - every object starts with a 4-byte magic and a u32 format version
//   - lengths are u64
inline constexpr std::uint32_t format_version = 1;

// Corrupt length fields must not trigger giant allocations, so bulk reads
// grow their buffer in chunks of this many elements. A truncated file then
// fails with an EOF error long before memory runs out.
inline constexpr std::size_t read_chunk = std::size_t{1} << 20;

class Writer {
public:
    explicit Writer(std::ostream& out) : out_(out) {}

    void bytes(const void* p, std::size_t n) {
        out_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        if (!out_) throw SerializationError("write failed");
    }

    template <class T>
        requires std::is_integral_v<T>
    void u(T v) {
        using U = std::make_unsigned_t<T>;
        auto x = static_cast<U>(v);
        std::array<unsigned char, sizeof(T)> buf{};
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            buf[i] = static_cast<unsigned char>(x & 0xFFu);
            if constexpr (sizeof(T) > 1) x = static_cast<U>(x >> 8);
        }
        bytes(buf.data(), buf.size());
    }

    void f32(float v) { u(std::bit_cast<std::uint32_t>(v)); }
    void f64(double v) { u(std::bit_cast<std::uint64_t>(v)); }
    void boolean(bool v) { u(static_cast<std::uint8_t>(v ? 1 : 0)); }
    void len(std::size_t n) { u(static_cast<std::uint64_t>(n)); }

    void str(const std::string& s) {
        len(s.size());
        bytes(s.data(), s.size());
    }

    void header(const char (&magic)[5]) {
        bytes(magic, 4);
        u(format_version);
    }

    void metric(Metric m) { u(static_cast<std::uint8_t>(m)); }

    void floats(const std::vector<float>& v) {
        len(v.size());
        if constexpr (std::endian::native == std::endian::little) {
            bytes(v.data(), v.size() * sizeof(float));
        } else {
            for (float x : v) f32(x);
        }
    }

    template <class T>
        requires std::is_integral_v<T>
    void ints(const std::vector<T>& v) {
        len(v.size());
        if constexpr (sizeof(T) == 1 || std::endian::native == std::endian::little) {
            bytes(v.data(), v.size() * sizeof(T));
        } else {
            for (T x : v) u(x);
        }
    }

    void metadata(const Metadata& m) {
        len(m.size());
        for (const auto& [key, value] : m) {
            str(key);
            u(static_cast<std::uint8_t>(value.index()));
            std::visit(
                [this](const auto& x) {
                    using X = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<X, std::int64_t>) u(x);
                    else if constexpr (std::is_same_v<X, double>) f64(x);
                    else if constexpr (std::is_same_v<X, bool>) boolean(x);
                    else str(x);
                },
                value);
        }
    }

private:
    std::ostream& out_;
};

class Reader {
public:
    explicit Reader(std::istream& in) : in_(in) {}

    void bytes(void* p, std::size_t n) {
        in_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in_.gcount()) != n) {
            throw SerializationError("unexpected end of data");
        }
    }

    template <class T>
        requires std::is_integral_v<T>
    T u() {
        using U = std::make_unsigned_t<T>;
        std::array<unsigned char, sizeof(T)> buf{};
        bytes(buf.data(), buf.size());
        U x = 0;
        for (std::size_t i = sizeof(T); i-- > 0;) {
            if constexpr (sizeof(T) > 1) x = static_cast<U>(x << 8);
            x = static_cast<U>(x | buf[i]);
        }
        return static_cast<T>(x);
    }

    float f32() { return std::bit_cast<float>(u<std::uint32_t>()); }
    double f64() { return std::bit_cast<double>(u<std::uint64_t>()); }

    bool boolean() {
        const auto b = u<std::uint8_t>();
        if (b > 1) throw SerializationError("invalid bool");
        return b == 1;
    }

    std::size_t len() {
        const auto n = u<std::uint64_t>();
        if (n > std::numeric_limits<std::size_t>::max()) {
            throw SerializationError("length does not fit in size_t");
        }
        return static_cast<std::size_t>(n);
    }

    std::string str() {
        const std::size_t n = len();
        std::string s;
        for (std::size_t done = 0; done < n;) {
            const std::size_t step = std::min(read_chunk, n - done);
            s.resize(done + step);
            bytes(s.data() + done, step);
            done += step;
        }
        return s;
    }

    void header(const char (&magic)[5], const char* what) {
        char got[4];
        bytes(got, 4);
        if (!std::equal(got, got + 4, magic)) {
            throw SerializationError(std::string(what) + ": bad magic (not a " + what +
                                     " file?)");
        }
        const auto v = u<std::uint32_t>();
        if (v != format_version) {
            throw SerializationError(std::string(what) + ": unsupported format version " +
                                     std::to_string(v));
        }
    }

    Metric metric() {
        const auto m = u<std::uint8_t>();
        if (m > static_cast<std::uint8_t>(Metric::Cosine)) {
            throw SerializationError("unknown metric " + std::to_string(m));
        }
        return static_cast<Metric>(m);
    }

    std::vector<float> floats() {
        const std::size_t n = len();
        std::vector<float> v;
        for (std::size_t done = 0; done < n;) {
            const std::size_t step = std::min(read_chunk, n - done);
            v.resize(done + step);
            if constexpr (std::endian::native == std::endian::little) {
                bytes(v.data() + done, step * sizeof(float));
            } else {
                for (std::size_t i = 0; i < step; ++i) v[done + i] = f32();
            }
            done += step;
        }
        return v;
    }

    template <class T>
        requires std::is_integral_v<T>
    std::vector<T> ints() {
        const std::size_t n = len();
        std::vector<T> v;
        for (std::size_t done = 0; done < n;) {
            const std::size_t step = std::min(read_chunk, n - done);
            v.resize(done + step);
            if constexpr (sizeof(T) == 1 || std::endian::native == std::endian::little) {
                bytes(v.data() + done, step * sizeof(T));
            } else {
                for (std::size_t i = 0; i < step; ++i) v[done + i] = u<T>();
            }
            done += step;
        }
        return v;
    }

    Metadata metadata() {
        Metadata m;
        const std::size_t n = len();
        for (std::size_t i = 0; i < n; ++i) {
            std::string key = str();
            MetaValue value;
            switch (u<std::uint8_t>()) {
                case 0: value = u<std::int64_t>(); break;
                case 1: value = f64(); break;
                case 2: value = boolean(); break;
                case 3: value = str(); break;
                default: throw SerializationError("unknown metadata type tag");
            }
            if (!m.emplace(std::move(key), std::move(value)).second) {
                throw SerializationError("duplicate metadata key");
            }
        }
        return m;
    }

private:
    std::istream& in_;
};

// Writes via `save(stream)` to `path`. Goes through a temp file and a rename,
// so a crash mid-save never leaves a half-written file at `path`.
template <class SaveFn>
void save_to_file(const std::filesystem::path& path, SaveFn&& save) {
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw SerializationError("cannot open " + tmp.string() + " for writing");
        try {
            save(out);
            out.flush();
            if (!out) throw SerializationError("write failed: " + tmp.string());
        } catch (...) {
            out.close();
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            throw;
        }
    }
    std::filesystem::rename(tmp, path);
}

template <class LoadFn>
auto load_from_file(const std::filesystem::path& path, LoadFn&& load) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw SerializationError("cannot open " + path.string() + " for reading");
    auto result = load(in);
    if (in.peek() != std::char_traits<char>::eof()) {
        throw SerializationError("trailing data after object in " + path.string());
    }
    return result;
}

}  // namespace io
}  // namespace overveckill
