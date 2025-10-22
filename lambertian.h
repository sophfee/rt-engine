#pragma once
#include "common.h"
#include "material.h"

struct hit_record;

class lambertian : public material {
public:
	lambertian(const color& albedo, const f64 metallic) : albedo(albedo), metallic(metallic) {}

	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override;

private:
	color albedo;
	f64 metallic;
};
