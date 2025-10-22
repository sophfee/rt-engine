// ReSharper disable CppClangTidyPerformanceUnnecessaryValueParam
// ReSharper disable CppPassValueParameterByConstReference
#pragma once

#include <vector>

#include "common.h"
#include "interval.h"

class material;
struct hit_record {
	p3d point;
	v3 normal;
	f64 u,v;
	std::double_t t;
	bool front_face;
	shared_ptr<material> mat;

	void set_face_normal(const ray& r, const v3& outward_normal) {
		// Sets the hit record normal vector.
		// NOTE: the parameter `outward_normal` is assumed to have unit length.

		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	virtual ~hittable() = default;
	virtual bool hit(const ray& r, interval ray_t, hit_record &rec) const = 0;
};

class hittable_list : public hittable {
public:
	vector<shared_ptr<hittable>> objects;

	hittable_list() = default;
	hittable_list(shared_ptr<hittable> object) : objects(1) { add(object); }

	void clear();
	void add(shared_ptr<hittable> object);
	bool hit(const ray &r, interval ray_t, hit_record &rec) const override;
};

inline void hittable_list::clear() { objects.clear(); }
inline void hittable_list::add(shared_ptr<hittable> object) { objects.push_back(object); }
inline bool hittable_list::hit(const ray &r, interval ray_t, hit_record &rec) const {
	hit_record temp;
	bool hit_any = false;
	auto closest = ray_t.max;
	for (const auto &object : objects) {
		if (object->hit(r, {ray_t.min, closest}, temp)) {
			hit_any = true;
			closest = temp.t;
			rec = temp;
		}
	}

	return hit_any;
}
