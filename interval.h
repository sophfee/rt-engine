#pragma once
#include "common.h"

struct interval {
	f64 min, max;

	interval() : min(DBL_MAX), max(-DBL_MAX) {}
	interval(const f64 start, const f64 end) : min(start), max(end) {}

	[[nodiscard]] f64 size() const;
	[[nodiscard]] bool contains(f64 x) const;
	[[nodiscard]] bool surrounds(f64 x) const;
	[[nodiscard]] f64 clamp(f64 x) const;
	[[nodiscard]] interval expand(f64 delta) const;

	static const interval empty, universe;
};
