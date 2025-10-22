#version 460 core
#extension GL_EXT_texture_array : enable
precision highp float;

/* -------------------- compilation options -------------------- */
#define TEMPORAL_SAMPLING
#define TEMPORAL_BUFFER_SIZE 64
#define DRAW_SPHERES 0
#define DRAW_QUADS 1
#define DRAW_BOXES 0
#define DRAW_SKY 0
/* -------------------- end -------------------- */

out vec4 FragColor;

const vec3 camera_center = vec3(0.0);
const vec3 camera_lookfrom = vec3(278.0, 278.0, -800.0);
const vec3 camera_lookat   = vec3(278.0, 278.0, 0.0);
const float focal_length = 1.0;
const uint MAX_PATH_LENGTH = 60u;

uniform vec4 noise;
layout(binding = 0, rgba8) coherent uniform image2DArray temporal0;


uniform int frames;
uniform float time;

in vec2 screenPosition;
in vec2 pixel_uv;

float saturate(float n) { return clamp(n, 0.0, 1.0); }
vec2  saturate(vec2 n)  { return clamp(n, vec2(0.0), vec2(1.0)); }
vec3  saturate(vec3 n)  { return clamp(n, vec3(0.0), vec3(1.0)); }
vec4  saturate(vec4 n)  { return clamp(n, vec4(0.0), vec4(1.0)); }

struct Ray { vec3 origin; vec3 normal; };
vec3 rayAt(in Ray r, float x) { return r.origin + x * r.normal; }

struct Interval { float min; float max; };
float intervalSize(inout Interval thus) { return thus.max - thus.min; }
bool intervalContains(inout Interval i, float x) { return (i.min <= x && x <= i.max); }
bool intervalSurrounds(inout Interval i, float x) { return (i.min < x && x < i.max); }

const float FLT_MAX = 3.40282346638528859812e+38;
const float EPSILON = 1e-3;
const float PI = 3.141592653589793;

uint xorshift32(inout uint state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}
float rand_f32(inout uint state) {
    state = xorshift32(state);
    // produce [0,1)
    return uintBitsToFloat(0x3f800000u | (state >> 9u)) - 1.0;
}

struct Material { vec3 color; vec3 emissive; bool specular; float ior; };
struct Sphere   { vec3 center; float radius; Material mat; };
struct Quad {
    vec3 Q;
    vec3 u;
    vec3 v;
    vec3 w;
    vec3 N;
    float D;
    Material mat;
};
struct Intersection {
    vec3 position;
    vec3 normal;
    float u;
    float v;
    float t;
    bool front_face;
    Material mat;
};
struct Scatter {
    vec3 attenuation;
    Ray ray;
    Material mat;
};

#define TWO_PI 6.2831853

vec3 sampleSphere(inout uint state) {
    float r0 = rand_f32(state);
    float r1 = rand_f32(state);
    float y = 1.0 - 2.0 * r0;
    float xz_r = sqrt(max(0.0, 1.0 - y * y));
    float phi = TWO_PI * r1;
    return vec3(xz_r * cos(phi), y, xz_r * sin(phi));
}
vec3 sample_lambertian(inout uint state, vec3 normal) {
    return normalize(normal + sampleSphere(state) * (1.0 - EPSILON));
}

Intersection failedIntersection() {
    Intersection o;
    o.position = vec3(0.0);
    o.normal = vec3(0.0);
    o.u = 0.0;
    o.v = 0.0;
    o.t = -1.0;
    o.front_face = false;
    o.mat.color = vec3(0.0);
    o.mat.emissive = vec3(0.0);
    o.mat.specular = false;
    o.mat.ior = 0.0;
    return o;
}

/* Sphere intersection */
Intersection intersectSphere(Ray ray, Sphere sphere) {
    vec3 v = ray.origin - sphere.center;
    float a = dot(ray.normal, ray.normal);
    float b = dot(v, ray.normal);
    float c = dot(v, v) - sphere.radius * sphere.radius;
    float d = b * b - a * c;
    if (d < 0.0) return failedIntersection();
    float sqrt_d = sqrt(d);
    float recip_a = 1.0 / a;
    float mb = -b;
    float t1 = (mb - sqrt_d) * recip_a;
    float t2 = (mb + sqrt_d) * recip_a;
    float t = (t1 > EPSILON) ? t1 : ((t2 > EPSILON) ? t2 : -1.0);
    if (t <= EPSILON) return failedIntersection();
    Intersection it;
    vec3 p = rayAt(ray, t);
    it.position = p;
    it.normal = (p - sphere.center) / sphere.radius;
    it.t = t;
    it.mat = sphere.mat;
    it.front_face = dot(ray.normal, it.normal) < 0.0;
    if (!it.front_face) it.normal = -it.normal;
    return it;
}

/* Quad helpers */
void setupQuad(inout Quad self) {
    vec3 n = cross(self.u, self.v);
    self.N = normalize(n);
    self.D = dot(self.N, self.Q);
    // w is a helper for calculating barycentric coords
    float nDotN = dot(self.N, self.N);
    self.w = (nDotN == 0.0) ? vec3(0.0) : self.N / nDotN;
}

bool quadIsInterior(Quad qd, float a, float b, inout Intersection hit) {
    Interval unit; unit.min = 0.0; unit.max = 1.0;
    if (!intervalContains(unit, a) || !intervalContains(unit, b)) return false;
    hit.u = a; hit.v = b; return true;
}

Intersection intersectQuad(Ray r, Quad qd) {
    Intersection rec = failedIntersection();
    float denom = dot(qd.N, r.normal);
    if (abs(denom) < 1e-8) return failedIntersection();
    float t = (qd.D - dot(qd.N, r.origin)) / denom;
    if (t <= EPSILON) return failedIntersection();
    vec3 it = rayAt(r, t);
    vec3 planarHitPtVector = it - qd.Q;
    float alpha = dot(qd.w, cross(planarHitPtVector, qd.v));
    float beta  = dot(qd.w, cross(qd.u, planarHitPtVector));
    if (!quadIsInterior(qd, alpha, beta, rec)) return failedIntersection();
    rec.t = t;
    rec.position = it;
    rec.front_face = dot(r.normal, qd.N) < 0.0;
    rec.normal = rec.front_face ? qd.N : -qd.N;
    rec.mat = qd.mat;
    return rec;
}

Quad make_quad(vec3 origin, vec3 u, vec3 v, Material mat) {
    Quad q;
    q.Q = origin;
    q.u = u;
    q.v = v;
    setupQuad(q);
    q.mat = mat;
    return q;
}

/* Box = 6 quads */
struct Box { Quad sides[6]; };

Box make_box(vec3 a, vec3 b, Material m) {
    vec3 minv = vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
    vec3 maxv = vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
    vec3 dx = vec3(maxv.x - minv.x, 0.0, 0.0);
    vec3 dy = vec3(0.0, maxv.y - minv.y, 0.0);
    vec3 dz = vec3(0.0, 0.0, maxv.z - minv.z);

    Box bX;
    // build each side explicitly to avoid array assignment
    bX.sides[0] = make_quad(vec3(minv.x, minv.y, maxv.z),  dx,  dy, m); // front
    bX.sides[1] = make_quad(vec3(maxv.x, minv.y, maxv.z), -dz,  dy, m); // right
    bX.sides[2] = make_quad(vec3(maxv.x, minv.y, minv.z), -dx,  dy, m); // back
    bX.sides[3] = make_quad(vec3(minv.x, minv.y, minv.z),  dz,  dy, m); // left
    bX.sides[4] = make_quad(vec3(minv.x, maxv.y, maxv.z),  dx, -dz, m); // top
    bX.sides[5] = make_quad(vec3(minv.x, minv.y, minv.z),  dx,  dz, m); // bottom

    return bX;
}

Intersection intersectBox(Ray r, Box b) {
    Intersection closest = failedIntersection();
    closest.t = FLT_MAX;
    for (int i = 0; i < 6; ++i) {
        Intersection hit1 = intersectQuad(r, b.sides[i]);
        if (hit1.t > 0.0 && hit1.t < closest.t) {
            closest = hit1;
        }
    }
    if (closest.t < FLT_MAX) return closest;
    return failedIntersection();
}

/* scattering and BSDF helpers */
float FresnelSchlick(float ior, float cos_theta) {
    float u = 1.0 - cos_theta;
    float sqrt_f0 = (ior - 1.0) / (ior + 1.0);
    float f0 = sqrt_f0 * sqrt_f0;
    return mix(f0, 1.0, u * u * u * u * u);
}

Scatter rayIntersectionScatter(inout uint state, Ray input_ray, Intersection hit) {
    vec3 I = normalize(input_ray.normal);
    float IdotN = dot(I, hit.normal);
    bool is_front_face = IdotN < 0.0;
    bool is_transmissive = hit.mat.ior > 0.0;
    float cos_theta = abs(IdotN);
    float ior = abs(hit.mat.ior);
    float ref_ratio = is_front_face ? (1.0 / ior) : ior;
    vec3 N = is_front_face ? hit.normal : -hit.normal;
    vec3 A = hit.mat.color;

    bool choose_specular = false;
    if (is_transmissive) {
        bool cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.0;
        float fr = FresnelSchlick(ref_ratio, cos_theta);
        choose_specular = cannot_refract || fr > rand_f32(state);
        if (choose_specular) A = vec3(1.0);
    } else {
        choose_specular = hit.mat.specular;
    }

    vec3 scattered;
    if (choose_specular) {
        scattered = reflect(I, N);
    } else if (is_transmissive) {
        scattered = refract(I, N, ref_ratio);
    } else {
        scattered = sample_lambertian(state, N);
    }

    Scatter sc;
    Ray scattered_ray;
    scattered_ray.origin = rayAt(input_ray, hit.t) + scattered * EPSILON; // push off surface
    scattered_ray.normal = normalize(scattered);
    sc.ray = scattered_ray;
    sc.attenuation = A;
    sc.mat = hit.mat;
    sc.mat.emissive = hit.mat.emissive;
    return sc;
}

vec3 skyColor(Ray ray) {
    float t = 0.5 * (normalize(ray.normal).y + 1.0);
    return mix(vec3(1.0), vec3(0.3, 0.5, 1.0), t);
}

/* scene: return closest hit (spheres, boxes, quads) */
Intersection scene(Ray r) {

    Intersection closest;
    closest.t = FLT_MAX;
    
#if DRAW_SPHERES
    // Spheres
    Sphere spheres[5];
    spheres[0].center = vec3(0.0, -101.5, -1.0);
    spheres[0].radius = 100.0;
    spheres[0].mat.color = vec3(0.6);
    spheres[0].mat.specular = false;
    spheres[0].mat.ior = 0.0;
    spheres[0].mat.emissive = vec3(0.0);

    float offset = 0.0;
    spheres[1].center = vec3(sin((time * 3.4) * PI) * 6.0, cos(offset), -10.0);
    spheres[1].radius = 0.8;
    spheres[1].mat.color = vec3(0.7, 0.4, 0.6);
    spheres[1].mat.specular = false;
    spheres[1].mat.ior = 0.0;
    spheres[1].mat.emissive = vec3(1.0, 0.0, 1.0);

    offset = 0.333333 * TWO_PI;
    spheres[2].center = vec3(cos((time * PI) + offset), sin((time * PI) + offset), -2.0);
    spheres[2].radius = 0.5;
    spheres[2].mat.color = vec3(0.5, 0.4, 0.0);
    spheres[2].mat.specular = false;
    spheres[2].mat.ior = 0.0;
    spheres[2].mat.emissive = vec3(0.0);

    offset = 0.666666 * TWO_PI;
    spheres[3].center = vec3(cos((time * PI) + offset), sin((time * PI) + offset), -2.0);
    spheres[3].radius = 0.5;
    spheres[3].mat.color = vec3(0.2, 0.5, 0.2);
    spheres[3].mat.specular = true;
    spheres[3].mat.ior = 0.0;
    spheres[3].mat.emissive = vec3(0.0);

    offset = 1.0000 * TWO_PI;
    spheres[4].center = vec3(cos((time * PI) + offset), sin((time * PI) + offset), -2.0);
    spheres[4].radius = 0.5;
    spheres[4].mat.color = vec3(0.7, 0.2, 0.7);
    spheres[4].mat.specular = true;
    spheres[4].mat.ior = 0.0;
    spheres[4].mat.emissive = vec3(1.0);

    // test spheres
    for (int i = 0; i < 5; i++) {
        Intersection hit1 = intersectSphere(r, spheres[i]);
        if (hit1.t > 0.0 && hit1.t < closest.t) {
            closest = hit1;
        }
    }
#endif

    // Materials for room/quads
    Material red;   red.color = vec3(.65, .05, .05); red.specular = false; red.ior = 0.0; red.emissive = vec3(0.0);
    Material white; white.color = vec3(.73);          white.specular = false; white.ior = 0.0; white.emissive = vec3(0.0);
    Material green; green.color = vec3(.12, .45, .15); green.specular = false; green.ior = 0.0; green.emissive = vec3(0.0);
    Material light; light.color = vec3(150.0);        light.specular = false; light.ior = 0.0; light.emissive = vec3(10000.0);
    
#if DRAW_BOXES
    // Boxes
    Box boxes[6];
    boxes[0] = make_box(vec3(0.49,-1.0,2.0), vec3(.5,1.0,1.0), green);
    boxes[1] = make_box(vec3(-1.0,-1.0,-3.0), vec3(1.0, 1.0, -1.0), light);
    // tweak faces for coloring
    boxes[1].sides[2].mat = red;
    boxes[1].sides[4].mat = green;
    boxes[2] = make_box(vec3(-0.5,-1.0,2.0), vec3(-0.49,1.0,1.0), red);
    boxes[3] = make_box(vec3( 0.05,1.0,0.30), vec3(-0.05, 2.0, 0.2), light);
    boxes[4] = make_box(vec3(-0.5, 0.0, 5.0), vec3(0.5, 0.0, 5.50), white);
    boxes[5] = make_box(vec3(-0.5,-3.950,1.0), vec3(0.5,-4.0, 5.0), white);
    
    // intersect boxes (all their sides)
    for (int b = 0; b < 6; ++b) {
        Intersection hb = intersectBox(r, boxes[b]);
        if (hb.t > 0.0 && hb.t < closest.t) closest = hb;
    }
#endif
    
#if DRAW_QUADS
    // Quads (Cornell-box)
    Quad quad[6];
    quad[0] = make_quad(vec3(555.0, 0.0,   0.0), vec3(0.0, 555.0, 0.0), vec3(0.0, 0.0, 555.0), green); // right wall
    quad[1] = make_quad(vec3(0.0,   0.0,   0.0), vec3(0.0, 555.0, 0.0), vec3(0.0, 0.0, 555.0), red);   // left wall
    quad[2] = make_quad(vec3(343.0, 554.0, 332.0), vec3(-130.0, 0.0, 0.0), vec3(0.0, 0.0, -105.0), light); // light
    quad[3] = make_quad(vec3(0.0,   0.0,   0.0), vec3(555.0, 0.0, 0.0), vec3(0.0, 0.0, 555.0), white);  // floor
    quad[4] = make_quad(vec3(555.0, 555.0, 555.0), vec3(-555.0, 0.0, 0.0), vec3(0.0, 0.0, -555.0), white); // ceiling
    quad[5] = make_quad(vec3(0.0,   0.0,   555.0), vec3(555.0, 0.0, 0.0), vec3(0.0, 555.0, 0.0), white);   // back wall

    // intersect quads
    for (int i = 0; i < 6; i++) {
        Intersection hit1 = intersectQuad(r, quad[i]);
        if (hit1.t > 0.0 && hit1.t < closest.t) closest = hit1;
    }
#endif

    if (closest.t < FLT_MAX) return closest;
    return failedIntersection();
}

bool isIntersectionValid(Intersection i) { return i.t > 0.0; }

/* A tiny bicubic sampler & jenkins hash */
vec4 cubic(float v){
    vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
    vec4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0 * s.x;
    float z = s.z - 4.0 * s.y + 6.0 * s.x;
    float w = 6.0 - x - y - z;
    return vec4(x, y, z, w) * (1.0/6.0);
}
uint jenkins_hash(uint i) {
    uint x = i;
    x += x << 10u;
    x ^= x >> 6u;
    x += x << 3u;
    x ^= x >> 11u;
    x += x << 15u;
    return x;
}

/* -------------------- main -------------------- */
void main() {
    // ---- Camera parameters ----
    vec3 vup = vec3(0.0, 1.0, 0.0);
    float vfov = radians(40.0);
    float aspect_ratio = 16.0 / 9.0;
    float viewport_height = 2.0 * tan(vfov / 2.0);
    float viewport_width  = aspect_ratio * viewport_height;
    float focus_dist = length(camera_lookfrom - camera_lookat);

    // ---- Camera basis ----
    vec3 w = normalize(camera_lookfrom - camera_lookat);
    vec3 u = normalize(cross(vup, w));
    vec3 v = cross(w, u);

    // ---- Image plane setup ----
    vec3 horizontal = focus_dist * viewport_width * u;
    vec3 vertical   = focus_dist * viewport_height * v;
    vec3 lower_left_corner =
    camera_lookfrom - horizontal * 0.5 - vertical * 0.5 - focus_dist * w;

    // ---- Pixel coordinate & jitter ----
    vec2 resolution = vec2(1920.0, 1080.0);
    vec2 pixelUV = gl_FragCoord.xy / resolution; // normalized [0,1]
    uvec2 pixel = uvec2(gl_FragCoord.xy);

    // create RNG state from pixel + noise
    uint seed = jenkins_hash((pixel.x ^ pixel.y) ^ jenkins_hash(floatBitsToUint(noise.w + noise.x + noise.y + noise.z)));
    // rand_f32 expects inout state, so keep it mutable
    uint state = seed;

    vec2 offset = vec2(rand_f32(state) - 0.5, rand_f32(state) - 0.5);
    offset /= resolution;
    vec2 uv = pixelUV + offset;

    // ---- Construct the ray (pinhole) ----
    Ray r;
    r.origin = camera_lookfrom;
    r.normal = normalize(lower_left_corner + uv.x * horizontal + uv.y * vertical - camera_lookfrom);

    // Path tracing loop (very simple)
    vec3 throughput = vec3(float(DRAW_SKY)); // cheeky way to setup this easily, if we draw the sky we have 1.0 as the base throughput, and 0.0 if we aren't drawing the sky (accumulates according to emissive components)
    vec3 radiance_sample = vec3(0.0);
    uint path_length = 0u;

    while (path_length < MAX_PATH_LENGTH) {
        Intersection hit = scene(r);
        if (!isIntersectionValid(hit)) {
#if DRAW_SKY
            radiance_sample += skyColor(r) * throughput;
            break;
#else
            radiance_sample += throughput;
#endif
        }
        // if the hit has emissive component, accumulate it and terminate optionally
        if (length(hit.mat.emissive) > 0.0) {
            radiance_sample += throughput * hit.mat.emissive;
            //break;
        }

        Scatter scattered = rayIntersectionScatter(state, r, hit);
        throughput *= scattered.attenuation;
        r = scattered.ray;
        path_length += 1u;

        // if throughput gets tiny, break
        if (max(throughput.r, max(throughput.g, throughput.b)) < 1e-4) break;
    }

    // Temporal accumulation
#ifdef TEMPORAL_SAMPLING
    ivec2 coords = ivec2(((screenPosition * 0.5 + 0.5)) * resolution);

    vec4 accumulated = vec4(radiance_sample, 1.0);
    vec4 temporalSample[TEMPORAL_BUFFER_SIZE];
    int presentCount = min(frames, TEMPORAL_BUFFER_SIZE);
    for (int i = 0; i < presentCount; i++) {
        int layer = (frames + i + 1) % TEMPORAL_BUFFER_SIZE;
        temporalSample[i] = imageLoad(temporal0, ivec3(coords, layer));
        accumulated += temporalSample[i];
    }

    vec3 color = pow(accumulated.rgb / float(max(1, presentCount)), vec3(1.0 / 2.2));

    // write back (rotate buffer)
    for (int i = 0; i < min(frames + 1, TEMPORAL_BUFFER_SIZE); i++) {
        int layer = (frames + i + 1) % TEMPORAL_BUFFER_SIZE;
        // write previous samples back (preserve)
        imageStore(temporal0, ivec3(coords, layer), temporalSample[i % TEMPORAL_BUFFER_SIZE]);
    }
    imageStore(temporal0, ivec3(coords, (frames % TEMPORAL_BUFFER_SIZE)), vec4(radiance_sample, 1.0));

    FragColor = vec4(color, 1.0);
#else
    FragColor = vec4(pow(radiance_sample, vec3(1.0 / 2.2)), 1.0);
#endif
}
