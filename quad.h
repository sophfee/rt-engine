// ReSharper disable CppPassValueParameterByConstReference
#pragma once
#include "hittable.h"

class quad : public hittable {
public:

	quad(const p3d& Q, const v3 &u, const v3 &v, shared_ptr<material> mat)  // NOLINT(modernize-pass-by-value)
		:  Q(Q), u(u), v(v), mat(mat) {
		auto n = cross(u, v);
		normal = unit_vector(n);
		D = dot(normal, Q);
	}
	bool hit(const ray &r, interval ray_t, hit_record &rec) const override;
	virtual bool is_interior(double a, double b, hit_record& rec) const {
		interval unit_interval = interval(0, 1);
		// Given the hit point in plane coordinates, return false if it is outside the
		// primitive, otherwise set the hit record UV coordinates and return true.

		if (!unit_interval.contains(a) || !unit_interval.contains(b))
			return false;

		rec.u = a;
		rec.v = b;
		return true;
	}

private:
	p3d Q;
	v3 u, v, w;
	v3 normal;
	f64 D;
	shared_ptr<material> mat;
};
