#include "metal.h"

#include "hittable.h"

bool metal::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
	const v3 reflected = reflect(r_in.direction(), rec.normal);
	scattered = ray(rec.point, reflected);
	attenuation = albedo;
	return true;
}
