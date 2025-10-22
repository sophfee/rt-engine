#version 460 core
#extension GL_EXT_texture_array : enable
precision highp float;

out vec4 FragColor;

const vec3 camera_center = vec3(0., 0., -0.);
const float focal_length = 1.50;
const uint MAX_PATH_LENGTH = 6u;

uniform vec4 noise;
uniform layout(binding=0,rgba8) coherent image2DArray temporal0;
// uniform sampler2DArray temporalSampler0;

#define TEMPORAL_BUFFER_SIZE 64

uniform int frames;
uniform float time;

in vec2 screenPosition;
in vec2 pixel_uv;

struct Ray {
    vec3 origin;
    vec3 normal;
};

vec3 rayAt(in Ray r, float x) {
    return r.origin + x * r.normal;
}

struct Interval {
    float min;
    float max;
};

float intervalSize(inout Interval thus) {
    return thus.max - thus.min;
}

bool intervalContains(inout Interval i, float x) {
    return (i.min <= x && x <= i.max);
}

bool intervalSurrounds(inout Interval i, float x) {
    return (i.min < x && x < i.max);
}

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
    // 0x3f800000u is the bit pattern for 1.0f
    // The mantissa bits come from the random bits, giving a float in [1.0, 2.0)
    // Subtract 1.0 to make it [0.0, 1.0)
    state = xorshift32(state);
    return uintBitsToFloat(0x3f800000u | (state >> 9u)) - 1.0;
}

/*

RayResult rayResultSetFaceNormal(RayResult result, Ray r, vec3 outwardNormal) {
    result.front_face = dot(r.normal, outwardNormal) < 0.0;
    result.normal = result.front_face ? outwardNormal : -outwardNormal;
    return result;
}



bool sphere(vec3 center, float radius, inout Interval i, inout Ray r, inout RayResult rec) {
    vec3 oc = center - r.origin;
    float a = length(r.normal)*length(r.normal);
    float h = dot(r.normal, oc);
    float c = length(oc) - radius;
    c *= c;
    float discriminant = h*h - a*c;
    if (discriminant < 0.0) return false;
    float sqrtd = sqrt(discriminant);
    
    float root = (h - sqrtd) / a;
    if (!intervalSurrounds(i, root))  {
        root = (h + sqrtd) / a;
        if (!intervalSurrounds(i, root))  {
            return false;
        }
    }
    
    rec.t = root;
    rec.position = rayAt(r, root);
    vec3 outward = (rec.position - center) / radius;
    rec = rayResultSetFaceNormal(rec, r, outward);
    return true;
}
*/

struct Material {
    vec3 color;
    vec3 emissive;
    bool specular;
    float ior;
};

struct Sphere {
    vec3 center;
    float radius;
    Material mat;
};

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

//Normal distribution function (Trowbridge-Reitz GGX)
float DistributionGGX(float NdotH, float alpha2)
{
    float f = (NdotH * alpha2 - NdotH) * NdotH + 1.0f;
    return alpha2 / (f * f * PI);
}

//Height-correlated Smith-GGX visibility function
float VisibilitySmithGGX(float NdotL, float NdotV, float alpha2)
{
    float SchlickGGX_V = NdotL * sqrt((-NdotV * alpha2 + NdotV) * NdotV + alpha2);
    float SchlickGGX_L = NdotV * sqrt((-NdotL * alpha2 + NdotL) * NdotL + alpha2);
    return 0.5f / (SchlickGGX_V + SchlickGGX_L);
}

//Approximated height-correlated Smith-GGX visibility function
float FastVisibilitySmithGGX(float NdotL, float NdotV, float alpha)
{
    return 0.5f / mix(2.0f * NdotL * NdotV, NdotL + NdotV, alpha);
}

//Schlick Fresnel function
vec3 FresnelSchlick(float cosTheta, vec3 f0, float f90)
{
    //float factor = pow(1.0f - cosTheta, 5);
    float factor  = 1.0f - cosTheta;
    float factor2 = factor * factor;
    factor        = factor2 * factor2 * factor;
    return f0 + (vec3(f90) - f0) * factor;
}

//Schlick Fresnel function with f90 = 1
vec3 FresnelSchlick(float cosTheta, vec3 f0)
{
    //float factor = pow(1.0f - cosTheta, 5);
    float factor  = 1.0f - cosTheta;
    float factor2 = factor * factor;
    factor        = factor2 * factor2 * factor;
    return factor + f0 * (1.0f - factor);
}

float FresnelSchlick(float ior, float cos_theta) {
    float u = 1 - cos_theta;
    float sqrt_f0 = (ior - 1.) / (ior + 1.);
    float f0 = sqrt_f0 * sqrt_f0;
    return mix(f0, 1., u * u * u * u * u);
}

//https://seblagarde.wordpress.com/2011/08/17/hello-world/
//Schlick Fresnel function with injected roughness term
vec3 FresnelSchlickRoughness(float cosTheta, vec3 f0, float alpha)
{
    //float factor = pow(1.0f - cosTheta, 5);
    float factor  = 1.0f - cosTheta;
    float factor2 = factor * factor;
    factor        = factor2 * factor2 * factor;
    return f0 + (max(vec3(1.0f - alpha), f0) - f0) * factor;
}

vec3 point_on_ray(Ray ray, float t) {
    return ray.origin + t * ray.normal;
}

vec3 pointOnRay(Ray ray, float t) {
    return ray.origin + t * ray.normal;
}
Intersection failedIntersection() {
    Intersection o;
    o.position = vec3(0.0);
    o.normal = vec3(0.0);
    o.t = -1.0;
    o.front_face = false;
    return o;
}

#define select(a, b, c) ((c) ? (b) : (a))

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

const float TWO_PI = 6.2831853;

vec3 sampleSphere(inout uint state) {
    float r0 = rand_f32(state); // random(noise.xy * (gl_FragCoord.xy)); //smoothstep(-1., 1.,);
    float r1 = rand_f32(state); // random(noise.zw * (gl_FragCoord.xy)); //smoothstep(-1., 1., random(noise.zw * gl_FragCoord.xy));
    
    float y = 1. - 2. * r0;
    
    float xz_r = sqrt(1. - y * y);
    
    float phi = TWO_PI * r1;
    return vec3(xz_r * cos(phi), y, xz_r * sin(phi));
}

vec3 sample_lambertian(inout uint state, vec3 normal) {
    return normal + sampleSphere(state) * (1. - EPSILON);
}

Intersection intersectSphere(Ray ray, Sphere sphere) {
    vec3 v = ray.origin - sphere.center;
    float a = dot(ray.normal, ray.normal);
    float b = dot(v, ray.normal);
    float c = dot(v, v) - sphere.radius * sphere.radius;
    
    float d = b * b - a * c;
    if (d < 0.) {
        return failedIntersection();
    }
    float sqrt_d = sqrt(d);
    float recip_a = 1.0 / a;
    float mb = -b;
    float t1 = (mb - sqrt_d) * recip_a;
    float t2 = (mb + sqrt_d) * recip_a;
    float t = select(t2, t1, t1 > EPSILON);
    if (t <= EPSILON) {
        return failedIntersection();
    }
    Intersection it;
    vec3 p = pointOnRay(ray, t);
    //it.position =  (p - sphere.center) / sphere.radius;
    it.normal = (p - sphere.center) / sphere.radius;
    it.t = t;
    it.mat = sphere.mat;
    return it;
}

void setupQuad(inout Quad self) {
    // RTinWE uses odd cordinates system
    //self.Q = self.Q.yzx;
    //self.u = self.u.yzx;
    //self.v = self.v.yzx;
    //self.Q -= camera_center;
    //self.Q -= vec3(278., 278., -800.);
    //self.u -= vec3(400.);
    //self.v -= vec3(400.);
    //self.Q /= 100.;
    //self.u /= 800.;
    //self.v /= 800.;
    // self.u -= camera_center;
    // self.v -= camera_center;
    
    vec3 n = cross(self.u, self.v);
    self.N = normalize(n);
    self.D = dot(self.N, self.Q);
    self.w = self.N / dot(self.N, self.N);
}

bool quadIsInterior(Quad qd, float a, float b, inout Intersection hit)
{
    Interval unit;
    unit.min = 0.0;
    unit.max = 1.0;
    
    if (!intervalContains(unit, a) || !intervalContains(unit, b)) {
        return false;
    }
    
    hit.u = a;
    hit.v = b;
    return true;
}

Intersection intersectQuad(Ray r, Quad qd) {
    Intersection rec;
    
    float denom = dot(qd.N, r.normal);
    if (abs(denom) < 1e-8) {
        return failedIntersection();
    }
    
    float t = (qd.D - dot(qd.N, r.origin)) / denom;
    vec3 it = rayAt(r, t);
    vec3 planarHitPtVector = it - qd.Q;
    float alpha = dot(qd.w, cross(planarHitPtVector, qd.v));
    float beta = dot(qd.w, cross(qd.u, planarHitPtVector));
    
    if (!quadIsInterior(qd, alpha, beta, rec)) {
        return failedIntersection();
    }
    
    rec.t = t;
    //rec.position = it;
    rec.front_face = dot(r.normal, qd.N) < 0.0;
    rec.normal = rec.front_face ? qd.N : -qd.N;
    rec.mat = qd.mat;
    return rec;
}

Scatter rayIntersectionScatter(inout uint state, Ray input_ray, Intersection hit) {
    vec3 I = normalize(input_ray.normal);
    float IdotN = dot(I, hit.normal);
    bool is_front_face = IdotN < 0.;
    bool is_transmissive = hit.mat.ior > 0.;
    float cos_theta = abs(IdotN);
    float ior = abs(hit.mat.ior);
    float ref_ratio = select(ior, 1. / ior, is_front_face);
    vec3 N = select(-hit.normal, hit.normal, is_front_face);
    vec3 A = hit.mat.color;
    bool choose_specular;
    if (is_transmissive) {
        bool cannot_refract = ref_ratio * ref_ratio * (1.0 - cos_theta * cos_theta) > 1.;
        choose_specular = cannot_refract || FresnelSchlick(cos_theta, ref_ratio) > rand_f32(state);//random(noise.wx*gl_FragCoord.xy);
        if (choose_specular) {
            A = vec3(1.0);
        }
    }else{
        choose_specular = hit.mat.specular; // == 1;
    }
    
    vec3 scattered; // = hit.specular ? reflect(input_ray.normal , hit.normal) : sample_lambertian(hit.normal);//
    if (choose_specular) {
        scattered = reflect(I, N);
    } else if (is_transmissive) {
        scattered = refract(I, N, ref_ratio);
    } else {
        scattered = sample_lambertian(state, N);
    }
    
    Scatter sc;
    
    Ray scattered_ray;
    scattered_ray.origin = pointOnRay(input_ray, hit.t);
    scattered_ray.normal = scattered;
    
    sc.ray = scattered_ray;
    sc.attenuation = A;
    sc.mat.emissive = hit.mat.emissive;
    return sc;
}

vec3 skyColor(Ray ray) {
    float t = 0.5 * (normalize(ray.normal).y + 1.);
    return (1. - t) * vec3(1.) + t * vec3(0.3, 0.5, 1.0);
}

float saturate(float n) {
    return clamp(n, 0., 1.);
}

vec2 saturate(vec2 n) {
    return clamp(n, vec2(0.), vec2(1.));
}

vec3 saturate(vec3 n) {
    return clamp(n, vec3(0.), vec3(1.));
}

vec4 saturate(vec4 n) {
    return clamp(n, vec4(0.), vec4(1.));
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

struct Box {
    Quad sides[6];
};

Intersection intersectBox(Ray r, Box b) {
    Intersection closest;
    closest.t = FLT_MAX;

    for (int i = 0; i < 6; i++) {
        Intersection hit1 = intersectQuad(r, b.sides[i]);
        if (hit1.t < closest.t && hit1.t > 0.) {
            closest.normal = hit1.normal;
            closest.t = hit1.t;
            closest.mat = hit1.mat;
        }
    }
    
    if (closest.t < FLT_MAX)
        return closest;
    return failedIntersection();
}

Box make_box(vec3 a, vec3 b, Material m) {
    vec3 minv = vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
    vec3 maxv = vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));

    vec3 dx = vec3(maxv.x - minv.x, 0.0, 0.0);
    vec3 dy = vec3(0.0, maxv.y - minv.y, 0.0);
    vec3 dz = vec3(0.0, 0.0, maxv.z - minv.z);
    
    Quad sides[6];

    sides[0] = make_quad(vec3(minv.x, minv.y, maxv.z),  dx,  dy, m); // front
    sides[1] = make_quad(vec3(maxv.x, minv.y, maxv.z), -dz,  dy, m); // right
    sides[2] = make_quad(vec3(maxv.x, minv.y, minv.z), -dx,  dy, m); // back
    sides[3] = make_quad(vec3(minv.x, minv.y, minv.z),  dz,  dy, m); // left
    sides[4] = make_quad(vec3(minv.x, maxv.y, maxv.z),  dx, -dz, m); // top
    sides[5] = make_quad(vec3(minv.x, minv.y, minv.z),  dx,  dz, m); // bottom
    
    Box bX;
    bX.sides = sides;
    return bX;
}

Intersection scene(Ray r) {
    
    Sphere spheres[5];
    spheres[0].center = vec3(0.0, -101.5, -1.0);
    spheres[0].radius = 100.0;
    spheres[0].mat.color = vec3(0.6);
    spheres[0].mat.specular = false;
    spheres[0].mat.ior = 0.0;
    spheres[0].mat.emissive = vec3(0.0,0.0, 0.0);

    float offset = 0.0;
    spheres[1].center = vec3(sin((time * 3.4) * PI) * 6.0, cos(offset), -10.0);
    spheres[1].radius = 0.8;
    spheres[1].mat.color = vec3(0.7, 0.4, 0.6);
    spheres[1].mat.specular = false;
    spheres[1].mat.ior = 0.0;
    //spheres[1mat.].emissive = vec4(0.0);
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

    Intersection closest;
    closest.t = FLT_MAX;
    
    for (int i = 0; i < 0; i++) {
        Intersection hit1 = intersectSphere(r, spheres[i]);
        if (hit1.t < closest.t && hit1.t > 0.) {
            //closest.normal = hit1.normal;
            //closest.t = hit1.t;
            //closest.mat = hit1.mat;
        }
    }
    
    Material red;
    red.color = vec3(.65, .05, .05);
    red.specular = false;
    red.ior = 0.0;
    red.emissive = red.color * vec3(0.);
    
    Material white;
    white.color = vec3(.73);
    white.specular = false;
    white.ior = 0.0;
    white.emissive = vec3(0.);

    Material green;
    green.color = vec3(.12, .45, .15);
    green.specular = false;
    green.ior = 0.0;
    green.emissive = green.color * vec3(0.);
    
    Material light;
    light.color = vec3(150.);
    light.specular = false;
    light.ior = 0.0;
    light.emissive = vec3(1.);

    Box boxes[6];

    boxes[0] = make_box(vec3(0.49,-1.0,2.), vec3(.5,1.,1.), green);

    boxes[0] = make_box(vec3(-1.0,-1.0,-3.0), vec3(1.0, 1.0, -1.0), light);
    boxes[0].sides[2].mat = red;
    boxes[0].sides[4].mat = green;

    boxes[1] = make_box(vec3( -0.5,-1.0,2.), vec3(-0.49,1.,1.), red);
    boxes[2] = make_box(vec3( 0.05,1.0,.30), vec3(-0.05, 2.0, 0.2), light);
    boxes[3] = make_box(vec3(-0.5, 0.0, 5.0), vec3(0.5, 0.0, 5.50), white); // back wall
    boxes[4] = make_box(vec3( -0.5,-3.950,1.), vec3(0.5,-4.0, 5.), white); // floor
    boxes[5] = make_box(vec3( -.5, 2.02, 2.), vec3(.5, 3.0, 5.0), white); // ceiling

    for (int i = 0; i < 1; i++) {
        Intersection hit1 = intersectBox(r, boxes[i]);
        if (hit1.t < closest.t && hit1.t > 0.) {
            closest.normal = hit1.normal;
            closest.t = hit1.t;
            closest.mat = hit1.mat;
        }
    }

    if (closest.t < FLT_MAX) {
        return closest;
    }
    
    return failedIntersection();
}

bool isIntersectionValid(Intersection i) {
    return i.t > 0.0;
}

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
vec4 textureBicubic(sampler2D sampler, vec2 texCoords){

    vec2 texSize = textureSize(sampler, 0);
    vec2 invTexSize = 1.0 / texSize;

    texCoords = texCoords * texSize - 0.5;


    vec2 fxy = fract(texCoords);
    texCoords -= fxy;

    vec4 xcubic = cubic(fxy.x);
    vec4 ycubic = cubic(fxy.y);

    vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

    vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

    offset *= invTexSize.xxyy;

    vec4 sample0 = texture(sampler, offset.xz);
    vec4 sample1 = texture(sampler, offset.yz);
    vec4 sample2 = texture(sampler, offset.xw);
    vec4 sample3 = texture(sampler, offset.yw);

    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);

    return mix(
    mix(sample3, sample2, sx), mix(sample1, sample0, sx)
    , sy);
}

void main() {
    //vec3 viewport_upper_left = camera_center - vec3(0, 0, focal_length) - 0.5;
    //vec3 pixel00_loc = viewport_upper_left + 0.5 * vec3(pixel_uv, 0.0);
    uvec2 pixel = uvec2(gl_FragCoord.xy * vec2(1920.0, 1080.0));
    uint state = jenkins_hash((pixel.x ^ pixel.y) ^ jenkins_hash(floatBitsToUint(noise.w + noise.x + noise.y + noise.z)));
    vec2 offset = vec2(rand_f32(state) - 0.5, rand_f32(state) - 0.5);
    offset *= vec2(1.0 / 1920.0, 1.0 / 1080.0);
    vec3 rd = normalize(vec3(pixel_uv + offset, -focal_length) - camera_center);
    
    
    Ray r;
    r.origin = camera_center;
    r.normal = rd;
    
    Interval v;
    v.min = 0.000;
    v.max = 10000.0;
    
    vec3 throughput = vec3(1.);
    vec3 radiance_sample = vec3(0.);
    
    float closest_t = FLT_MAX;
    uint path_length = 0u;
    
    while (path_length < MAX_PATH_LENGTH) {
        Intersection hit = scene(r);
        if (!isIntersectionValid(hit)) {
            // If no intersection was found, return the color of the sky and terminate the path.
            radiance_sample += path_length > 0 ? throughput : vec3(0.); //  * skyColor(r);
            break;
        }

        Scatter scattered = rayIntersectionScatter(state, r, hit);
        //throughput += (scattered.mat.emissive.rgb);
        throughput *= scattered.attenuation;
        r = scattered.ray;
        path_length += 1u;
    }
    #define TEMPORAL_SAMPLING
    #ifdef TEMPORAL_SAMPLING
    ivec2 coords = ivec2(((screenPosition * .5 + .5)) * vec2(1920.0, 1080.0));
    
    vec4 accumulated = vec4(radiance_sample, 1.0);
    vec4 temporalSample[TEMPORAL_BUFFER_SIZE];
    for (int i = 0; i < min(frames, TEMPORAL_BUFFER_SIZE); i++) {
        int layer = (frames + i + 1) % TEMPORAL_BUFFER_SIZE;
        temporalSample[i] = imageLoad(temporal0, ivec3(coords, layer)); // * 8.0/8.0;
        accumulated += temporalSample[i]; // * sqrt((float(TEMPORAL_BUFFER_SIZE - i) / float(TEMPORAL_BUFFER_SIZE)));
    }
    
    vec3 color = pow(accumulated.rgb / min(frames, TEMPORAL_BUFFER_SIZE), vec3(1. / 2.2));
    
    for (int i = 0; i < min(frames + 1, TEMPORAL_BUFFER_SIZE); i++) {
        int layer = (frames + i + 1) % TEMPORAL_BUFFER_SIZE;
        imageStore(temporal0, ivec3(coords, layer), temporalSample[i % TEMPORAL_BUFFER_SIZE]);//((frames + (i)) % TEMPORAL_BUFFER_SIZE)]);
    }
    imageStore(temporal0, ivec3(coords, ((frames) % TEMPORAL_BUFFER_SIZE)), vec4(radiance_sample.rgb, 1.0)); // vec4((accumulated.rgb / min(frames+1, 8)).rgb, 1.0));

    FragColor = vec4(color, 1.0);//closest_t < 1e37 ? vec4(vec3(saturate(closest_t)), 1.)  : vec4(skyColor(r), 1.0);
    #else
    FragColor = vec4(pow(radiance_sample.rgb, vec3(1. / 2.2)), 1.0);
    #endif
}