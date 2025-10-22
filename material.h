#pragma once
#include "common.h"

struct hit_record;
struct ray;
class material {
public:
	virtual ~material() = default;
	virtual [[nodiscard]] bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered) const;
};
