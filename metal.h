#pragma once
#include "material.h"

class metal : public material  {
public:
	metal(const color &albedo) : albedo(albedo) {}

	bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const override;
private:
	color albedo;
};
