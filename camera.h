#pragma once

#include <array>
#include <barrier>
#include <condition_variable>
#include <cstdint>
#include <optional>
#include <semaphore>
#include <thread>

#include "common.h"
#include "SDL2/Include/SDL.h"

class hittable;

// The camera class acts as a world space object that also owns a renderer and render target texture.
class camera {
public:
	f64 vertical_fov		= 40.0;
	f64 defocus_angle		= 0;  // Variation angle of rays through each pixel
	f64 focus_dist			= 10;    // Distance from camera lookfrom point to plane of perfect focus
	f64 aspect_ratio		= 1.0;
	i32 image_width			= 100;
	i32 samples_per_pixel	= 10;
	int max_depth			= 10;
	p3d lookfrom			= {+0,+0,+0};   // Point camera is looking from
	p3d lookat				= {+0,+0,-1};  // Point camera is looking at
	u64 frames				= 1; // the number of frames rendered. used for additive sampling.
	v3 vup					= {+0,+1,+0};     // Camera-relative "up" direction

	camera(SDL_Window *window, i32 width, i32 height);
	~camera();
	
	void render(const hittable &w);

private:
	
	//std::optional<std::reference_wrapper<const hittable>> world;
	i32 image_height;
	f64 pixel_samples_scale = 3.0;
	p3d center; // camera center
	p3d pixel00_loc; // location of 0,0
	v3 pixel_delta_u; // Offset to pixel to the right
	v3 pixel_delta_v; // offset to pixel below
	v3 u, v, w;              // Camera frame basis vectors
	v3 defocus_disk_u;       // Defocus disk horizontal radius
	v3 defocus_disk_v;       // Defocus disk vertical radius
	u32 *rendertarget;

	std::mutex render_buffer_write_mtx;
	std::array<std::jthread, 16> threads;
	std::array<std::condition_variable, 16> cv{std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable()};
	std::array<std::condition_variable, 16> lp{std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable(), std::condition_variable()};
	std::array<std::mutex, 16> mx{std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex(), std::mutex()};
	std::array<std::vector<std::int32_t>, 16> targets;

	SDL_Renderer *rnd = nullptr;
	SDL_Texture *tex = nullptr;
	
	void compute_row(const hittable &world, int j);

	void initialize();
	[[nodiscard]] ray get_ray(i32 i, i32 j) const;
	[[nodiscard]] color ray_color(const ray &r, i32 depth, const hittable &world) const;
	[[nodiscard]] static v3 sample_disk(f64 radius);
	[[nodiscard]] p3d defocus_disk_sample() const;
	static [[nodiscard]] v3 sample_square(); 
};
