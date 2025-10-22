// ReSharper disable CppTooWideScopeInitStatement
#include "camera.h"

#include <cassert>
#include <functional>
#include <future>
#include <iostream>

#include "hittable.h"
#include "material.h"
#include "ray.h"

camera::camera(SDL_Window *window, const i32 width, const i32 height) :
	center(0.0, 0.0, 1.0),
	rnd(SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED))
{
	assert(rnd != nullptr);
	this->image_width = width;
	this->image_height = height;
	this->aspect_ratio = static_cast<float>(image_width) / static_cast<float>(image_height);
	this->tex = SDL_CreateTexture(
		rnd,
		SDL_PIXELFORMAT_RGBA8888,
		SDL_TEXTUREACCESS_STREAMING,
		image_width,
		image_height
	);
	this->rendertarget = static_cast<u32 *>(std::malloc(sizeof(u32) * image_width * image_height));
	assert(rendertarget != nullptr);
/*
	for (int i = 0; i < 16; i++) {
		threads[i] = std::jthread([this, i] { compute_row(i); });
		targets[i] = {};
		for (int x = 0; x < (image_height % 16) + 1; x++) {
			targets[i].push_back(i * 16 + x);
		}
	}
*/
	this->pixel_samples_scale = 1.0 / samples_per_pixel;
}

camera::~camera() {
	SDL_DestroyTexture(this->tex);
	SDL_DestroyRenderer(this->rnd);
	std::free(this->rendertarget);
}

void camera::render(const hittable &w) {
	this->initialize();
	
	std::vector<std::future<void>> futures;
	for (int j = 0; j < image_height; j++) {
		futures.emplace_back(std::async([this, j, &w] { compute_row(w, j); }));
	}

	for (auto &future : futures) {
		future.get(); // waits
	}
	
	//for (int i = 0; i < 16; i++) {
	//	end[i].acquire();
	//

	SDL_UpdateTexture(tex, nullptr, rendertarget, sizeof(u32) * image_width);
	SDL_RenderCopy(rnd, tex, nullptr, nullptr);
	SDL_RenderPresent(rnd);
	//SDL_SetTextureBlendMode(tex, SDL_BLENDMODE_BLEND);
	frames++;
}
void camera::compute_row(const hittable &world, const int j) {
	auto temp = new u32[image_width];
	for (int i = 0; i < image_width; i++) {
		color pixel_color(0, 0, 0);
		for (int sample = 0; sample < samples_per_pixel; sample++) {
			ray r = get_ray(i, j);
			pixel_color += ray_color(r, max_depth, world);
		}
		u32 color_sdl = color_to_sdl(pixel_samples_scale * pixel_color);

		#ifdef ADAPTIVE_SAMPLING
		// remove alpha bit
		u32 old_value = rendertarget[image_width * j + i];// & 0xFFFFFF00;

		if (frames != 0xFFFFFFFF) {
			temp[i] = color_sdl;
		}
		else {
			u8 col[4]{};
			col[3] = color_sdl & 0xFF000000 >> 24;
			col[2] = color_sdl & 0x00FF0000 >> 16;
			col[1] = color_sdl & 0x0000FF00 >> 8;
			col[0] = color_sdl & 0x000000FF;
		
			f64 colf[4]{};
			colf[3] = static_cast<f64>(col[3]) / 256.0;
			colf[2] = static_cast<f64>(col[2]) / 256.0;
			colf[1] = static_cast<f64>(col[1]) / 256.0;
			colf[0] = static_cast<f64>(col[0]) / 256.0;
		
			u8 old[4]{};
			old[3] = old_value & 0xFF000000 >> 24;
			old[2] = old_value & 0x00FF0000 >> 16;
			old[1] = old_value & 0x0000FF00 >> 8;
			old[0] = old_value & 0x000000FF;

			f64 oldf[4]{};
			oldf[3] = static_cast<f64>(old[3]) / 256.0;
			oldf[2] = static_cast<f64>(old[2]) / 256.0;
			oldf[1] = static_cast<f64>(old[1]) / 256.0;
			oldf[0] = static_cast<f64>(old[0]) / 256.0;

			f64 newf[4]{};
			newf[3] = std::lerp(oldf[3], colf[3], 0.05);
			newf[2] = std::lerp(oldf[2], colf[2], 0.05);
			newf[1] = std::lerp(oldf[1], colf[1], 0.05);
			newf[0] = std::lerp(oldf[0], colf[0], 0.05);

			u32 output = static_cast<u32>(newf[3] * 256) << 24
			+ static_cast<u32>(newf[2] * 256) << 16
			+ static_cast<u32>(newf[1] * 256) << 8
			 + static_cast<u32>(newf[0] * 256);
		
			temp[i] = output;
		}
		#else
		
		temp[i] = color_sdl;
		
		#endif
	}

	{
		std::lock_guard lock(this->render_buffer_write_mtx);
		// offset by width * row index
		std::memcpy(&rendertarget[static_cast<size_t>(image_width * j)], temp, sizeof(u32) * image_width);
	}

	delete[] temp;
	
	/*
	while (true) {
		//std::unique_lock ul(mx[thread_id]);
		//cv[thread_id].wait(ul);
		for (const std::int32_t j : targets[thread_id]) {
			//std::cout << j << std::endl;
			
		}
		//lp[thread_id].notify_one();
	}
	*/
}
void camera::initialize() {
	image_height = static_cast<int>(image_width / aspect_ratio);
	image_height = (image_height < 1) ? 1 : image_height;

	center = lookfrom;

	// Determine viewport dimensions.
	const f64 theta = deg2rad(vertical_fov);
	const f64 h = std::tan(theta/2);
	const f64 viewport_height = 2 * h * focus_dist;
	const f64 viewport_width = viewport_height * (static_cast<f64>(image_width)/image_height);

	// Calculate the u,v,w unit basis vectors for the camera coordinate frame.
	w = unit_vector(lookfrom - lookat);
	u = unit_vector(cross(vup, w));
	v = cross(w, u);

	// Calculate the vectors across the horizontal and down the vertical viewport edges.
	const v3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
	const v3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

	// Calculate the horizontal and vertical delta vectors from pixel to pixel.
	pixel_delta_u = viewport_u / image_width;
	pixel_delta_v = viewport_v / image_height;

	// Calculate the location of the upper left pixel.
	const v3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
	pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

	// Calculate the camera defocus disk basis vectors.
	const f64 defocus_radius = focus_dist * std::tan(deg2rad(defocus_angle / 2));
	defocus_disk_u = u * defocus_radius;
	defocus_disk_v = v * defocus_radius;
}

ray camera::get_ray(const i32 i, const i32 j) const {
	const v3 offset = sample_square();
	const v3 pixel_sample = pixel00_loc
	                        + ((i + offset.x()) * pixel_delta_u)
	                        + ((j + offset.y()) * pixel_delta_v);

	p3d ray_origin = defocus_angle <= 0 ? center : defocus_disk_sample();
	v3 ray_direction = pixel_sample - ray_origin;

	return {ray_origin, ray_direction};
}

color camera::ray_color(const ray &r, const i32 depth, const hittable &world) const {
	if (depth <= 0)
		return {0.0, 0.0, 0.0};
	
	if (hit_record rec; world.hit(r, interval(0, std::numeric_limits<double>::infinity()), rec)) {
		v3 direction = random_on_hemisphere(rec.normal);
		ray scattered;
		color attenuation;
		if (rec.mat->scatter(r, rec, attenuation, scattered)) {
			return attenuation * ray_color(scattered, depth - 1, world);
		}
		return {0.0, 0.0, 0.0};
	}

	const v3 unit_direction = unit_vector(r.direction());
	const f64 a = 0.5*(unit_direction.y() + 1.0);
	return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

v3 camera::sample_disk(const f64 radius) {
	return radius * random_in_unit_disk();
}

p3d camera::defocus_disk_sample() const {
	// Returns a random point in the camera defocus disk.
	v3 p = random_in_unit_disk();
	return center + p[0] * defocus_disk_u + p[1] * defocus_disk_v;
}

v3 camera::sample_square() {
	return {random_double() - 0.5, random_double() - 0.5, 0};
}
