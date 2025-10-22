#version 460 core

layout (location = 0) in vec3 aPos;

out vec2 screenPosition;
out vec2 pixel_uv;

const float aspect_ratio = 16.0 / 9.0;

void main() {
    gl_Position = vec4(aPos, 1.0);
    screenPosition = gl_Position.xy;
    pixel_uv = (aPos.xy) * vec2(aspect_ratio, 1.0);
}