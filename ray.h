#pragma once

#include <complex>
#include <numbers>

#include "common.h" // ray is defined here as a struct, but this defines functionality to cast
#include "hittable.h"

inline double hit_sphere(const p3d& center, const double radius, const ray& r) {
	const v3 oc = center - r.origin();
	const double_t a = dot(r.direction(), r.direction());
	const double b = -2.0 * dot(r.direction(), oc);
	const double c = dot(oc, oc) - radius * radius;
	const double discriminant = b * b - 4 * a * c;
	if (discriminant < 0) {
		return -1.0;
	} else {
		return (-b - std::sqrt(discriminant) ) / (2.0*a);
	}
}

inline color ray_color(const ray& r,  hittable_list &world) {
	hit_record rec{};
	if (world.hit(r, {0, INFINITY}, rec)) {
		return 0.5 * (rec.normal + color(1,1,1));
	}

	v3 unit_direction = unit_vector(r.direction());
	auto a = 0.5*(unit_direction.y() + 1.0);
	return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

inline std::uint32_t color_to_sdl(v3 color) {
	std::uint8_t color_ptr[4]{};
	static const interval intensity(0.000, 0.999);
	color_ptr[3] = static_cast<std::uint8_t>(intensity.clamp(color.x()) * 256.0);
	color_ptr[2] = static_cast<std::uint8_t>( intensity.clamp(color.y()) * 256.0);
	color_ptr[1] = static_cast<std::uint8_t>( intensity.clamp(color.z()) * 256.0);
	color_ptr[0] = 0xFF;
	const std::uint32_t result = *reinterpret_cast<std::uint32_t*>(color_ptr);
	return result;
}