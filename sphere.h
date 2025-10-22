#pragma once

#include "hittable.h"

class material;

class sphere : public hittable {
public:

	sphere(const p3d& center, const std::double_t radius, shared_ptr<material> material) : center(center), radius(radius), mat(std::move(material)) {}
	bool hit(const ray &r, interval ray_t, hit_record &rec) const override;

private:
	p3d center;
	std::double_t radius = 0.0;
	shared_ptr<material> mat;
};

inline bool sphere::hit(const ray &r, const interval ray_t, hit_record &rec) const {
	const v3 oc = center - r.origin();
	const auto a = r.direction().length_squared();
	const auto h = dot(r.direction(), oc);
	const auto c = oc.length_squared() - radius*radius;

	const auto discriminant = h*h - a*c;
	if (discriminant < 0)
		return false;

	const auto sqrtd = std::sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range.
	auto root = (h - sqrtd) / a;
	if (!ray_t.surrounds(root)) {
		root = (h + sqrtd) / a;
		if (!ray_t.surrounds(root))
			return false;
	}

	rec.t = root;
	rec.point = r.at(rec.t);
	rec.mat = mat;
	const v3 outward = (rec.point - center) / radius;
	rec.set_face_normal(r, outward);

	return true;
}