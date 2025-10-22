#include "quad.h"

bool quad::hit(const ray &r, interval ray_t, hit_record &rec) const {
	auto denom = dot(normal, r.direction());

	// No hit if the ray is parallel to the plane.
	if (std::fabs(denom) < 1e-8)
		return false;

	// Return false if the hit point parameter t is outside the ray interval.
	auto t = (D - dot(normal, r.origin())) / denom;
	if (!ray_t.contains(t))
		return false;

	// Determine if the hit point lies within the planar shape using its plane coordinates.
	auto intersection = r.at(t);
	v3 planar_hitpt_vector = intersection - Q;
	auto alpha = dot(w, cross(planar_hitpt_vector, v));
	auto beta = dot(w, cross(u, planar_hitpt_vector));

	if (!is_interior(alpha, beta, rec))
		return false;

	// Ray hits the 2D shape; set the rest of the hit record and return true.
	rec.t = t;
	rec.point = intersection;
	rec.mat = mat;
	rec.set_face_normal(r, normal);

	return true;
}