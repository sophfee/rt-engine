# rt-engine
Learning Path Tracing techniques and such. If it works on my system that's all that really matters. Currently, only has software ray tracing, gpu path tracing, but no hardware ray tracing.

## Project Details
* C++20.
* GLAD for OpenGL bindings.
* SDL2 for windowing and context management.

## Software Renderer
* Uses SDL2 to draw.
* Renders to a single texture
* Multithreaded, initially with std::jthread, changes to std::async because frankly it's easier.
* Thread-safe mostly. No hard crashes yet.

## Path Tracer
* Uses OpenGL to draw.
* Full screen quad setup.
* I keep 99% of the code isolated on the GPU.
* Pure OpenGL code, no wrapper junk.
* Temporal Sampling, with adjustable frames to average from (default 64 because of how noisy path tracing is, but doesn't work great with motion)

## Hardware Raytracing
* TBD
