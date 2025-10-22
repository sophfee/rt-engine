#pragma once
#include <cmath>
#include <cstdint>
#include <numbers>
#include <ostream>
#include <random>
#include <vector>

using std::shared_ptr;
using std::unique_ptr;
using std::make_unique;
using std::make_shared;
using std::vector;

using f32 = std::float_t;
using f64 = std::double_t;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;


constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = std::numbers::pi;

inline double random_double() {
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

struct v3 {
    f64 e[3];

    v3() : e{0,0,0} {}
    v3(const f64 e0, const f64 e1, const f64 e2) : e{e0, e1, e2} {}

    [[nodiscard]] f64 x() const { return e[0]; }
    [[nodiscard]] f64 y() const { return e[1]; }
    [[nodiscard]] f64 z() const { return e[2]; }

    v3 operator-() const { return {-e[0], -e[1], -e[2]}; }
    f64 operator[](const std::size_t i) const { return e[i]; }
    f64& operator[](const std::size_t i) { return e[i]; }

    v3& operator+=(const v3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    v3& operator*=(const f64 t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    v3& operator/=(const f64 t) {
        return *this *= 1/t;
    }

    [[nodiscard]] f64 length() const {
        return std::sqrt(length_squared());
    }

    [[nodiscard]] f64 length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
    [[nodiscard]] bool near_zero() const;

    static v3 random() {
        return v3(random_double(), random_double(), random_double());
    }

    static v3 random(double min, double max) {
        return v3(random_double(min,max), random_double(min,max), random_double(min,max));
    }
};

inline bool v3::near_zero() const {
    auto s = 1e-8;
    return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
}

using p3d = v3;

inline std::ostream& operator<<(std::ostream& out, const v3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
inline v3 operator+(const v3& u, const v3& v) {
    return {u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]};
}
inline v3 operator-(const v3& u, const v3& v) {
    return {u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]};
}
inline v3 operator*(const v3& u, const v3& v) {
    return {u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]};
}
inline v3 operator*(const double t, const v3& v) {
    return {t*v.e[0], t*v.e[1], t*v.e[2]};
}
inline v3 operator*(const v3& v, const f64 t) {
    return t * v;
}
inline v3 operator/(const v3& v, const double t) {
    return 1/t * v;
}
inline f64 dot(const v3& u, const v3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}
inline v3 cross(const v3& u, const v3& v) {
    return {u.e[1] * v.e[2] - u.e[2] * v.e[1],
               u.e[2] * v.e[0] - u.e[0] * v.e[2],
               u.e[0] * v.e[1] - u.e[1] * v.e[0]};
}
inline v3 unit_vector(const v3& v) {
    return v / v.length();
}


inline v3 random_in_unit_disk() {
    while (true) {
        if (v3 p = {random_double(-1, 1), random_double(-1, 1), 0};
            p.length_squared() < 1)
            return p;
    }
}

inline v3 random_unit_vector() {
    while (true) {
        v3 p = v3::random(-1, 1);
        if (const f64 lensq = p.length_squared(); 1e-160 < lensq && lensq <= 1.0)
            return p / sqrt(lensq);
    }
}

inline v3 random_on_hemisphere(const v3& normal) {
    v3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

inline v3 reflect(const v3& v, const v3& n) {
    return v - 2 * dot(v, n) * n;
}

inline v3 refract(const v3 &uv, const v3 &n, const f64 etai_over_etat) {
    const f64 cos_theta = std::fmin(dot(-uv, n), 1.0);
    const v3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    const v3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

struct ray {
    ray() = default;
    ray(const p3d &origin, const v3 &normal) : orig(origin), norm(normal) {}

    [[nodiscard]] const p3d &origin() const { return orig; }
    [[nodiscard]] const p3d &normal() const { return norm; }
    [[nodiscard]] const p3d &direction() const { return norm; }

    [[nodiscard]] p3d at(const f64 t) const {
        return orig + t * norm;
    }
    
private:
    p3d orig;
    v3 norm;
};

using color = v3;



inline f64 deg2rad(const f64 degrees) {
    return degrees * std::numbers::pi_v<f64> / 180.0;
}