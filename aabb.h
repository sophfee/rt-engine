// ReSharper disable CppTooWideScopeInitStatement
#pragma once
#include <algorithm>

#include "interval.h"

// axis-aligned bounding box
struct aabb {
	interval x, y, z;
	aabb() {} // The default AABB is empty, since intervals are empty by default.

	aabb(const interval& x, const interval& y, const interval& z)
	  : x(x), y(y), z(z) {}

	aabb(const p3d& a, const p3d& b) {
		// Treat the two points a and b as extrema for the bounding box, so we don't require a
		// particular minimum/maximum coordinate order.

		x = (a[0] <= b[0]) ? interval(a[0], b[0]) : interval(b[0], a[0]);
		y = (a[1] <= b[1]) ? interval(a[1], b[1]) : interval(b[1], a[1]);
		z = (a[2] <= b[2]) ? interval(a[2], b[2]) : interval(b[2], a[2]);
	}

	[[nodiscard]] const interval& axis_interval(int n) const {
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	[[nodiscard]] bool hit(const ray& r, interval ray_t) const {
		const p3d& ray_orig = r.origin();
		const v3&   ray_dir  = r.direction();

		for (int axis = 0; axis < 3; axis++) {
			const interval& ax = axis_interval(axis);
			const f64 adinv = 1.0 / ray_dir[axis];

			const auto t0 = (ax.min - ray_orig[axis]) * adinv;
			const auto t1 = (ax.max - ray_orig[axis]) * adinv;

			if (t0 < t1) {
				ray_t.min = std::max(t0, ray_t.min);
				ray_t.max = std::min(t1, ray_t.max);
			} else {
				ray_t.min = std::max(t1, ray_t.min);
				ray_t.max = std::min(t0, ray_t.max);
			}

			if (ray_t.max <= ray_t.min)
				return false;
		}
		return true;
	}
};
