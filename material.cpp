#include "common.h"
#include "material.h"
#include "hittable.h"

bool material::scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const {
	return false;
}
