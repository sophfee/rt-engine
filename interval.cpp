#include "interval.h"

const interval interval::empty = interval(+DBL_MAX, -DBL_MAX);
const interval interval::universe = interval(-DBL_MAX, +DBL_MAX);

f64 interval::size() const {
	return max - min;
}

bool interval::contains(const f64 x) const {
	return min <= x && x <= max;
}

bool interval::surrounds(const f64 x) const {
	return min < x && x < max;
}

f64 interval::clamp(const f64 x) const {
	if (x < min) return min;
	if (x > max) return max;
	return x;
}

interval interval::expand(const f64 delta) const {
	const f64 padding = delta / 2;
	return {min + padding, max + padding};
}
