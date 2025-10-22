#include <cassert>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>

#include "camera.h"
#include "GLFW/glfw3.h"
#include "SDL2/Include/SDL.h"

#include "common.h"
#include "hittable.h"
#include "lambertian.h"
#include "material.h"
#include "metal.h"
#include "quad.h"
#include "sphere.h"
#include "ray.h"
#include "glad/glad.h"

#define rt_assert(COND)                             \
    if (!(COND)){SDL_Quit();return EXIT_FAILURE;}   \

// #define SOFTWARE_RENDER

int main(int argc, char* argv[]) {
    if (SDL_InitSubSystem(SDL_INIT_EVERYTHING) < 0) {
        return EXIT_FAILURE;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    //SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    const std::double_t aspect_ratio = 16.0 / 9.0;
    const std::int32_t image_width = 1920/ 16;

    std::int32_t image_height = static_cast<std::int32_t>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    
    SDL_Window *win = SDL_CreateWindow(
        "Ray tracing",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        1920, 1080,
        SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL
    );
    rt_assert(win != nullptr)
    
#ifdef SOFTWARE_RENDER
    
    const auto cam = new camera(win, image_width, image_height);
    //cam->lookfrom = {0, 0, -1.5};
    
    hittable_list world;
    //world.add(make_shared<sphere>(p3d{0.0, 1.0, -2.0}, 1.0));
    //world.add(make_shared<sphere>(p3d{0.50, 0.0, -2.0}, 1.0));

    auto red = make_shared<lambertian>(color(1.0, 0.0, 0.0), 1.0);
    auto blue = make_shared<lambertian>(color(0.0, 1.0, 1.0), 1.0);
    auto metal = make_shared<::metal>(color(1.0, 1.0, 1.0));
    
    world.add(make_shared<sphere>(p3d(0,0,-1), 0.5, red));
    world.add(make_shared<sphere>(p3d(0,-100.5,-1), 100, blue));
    world.add(make_shared<sphere>(p3d(-1.2, 0, -1), .35, metal));
    world.add(make_shared<sphere>(p3d(1.2, 0, -1), .15, metal));
    world.add(make_shared<quad>(p3d(5,0,0), v3(0,5,0), v3(0,0,5), blue));

    SDL_Event e;
    bool quit = false;
    while (win != nullptr && !quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

    cam->render(world);
    }
    
    delete cam;
#else
    SDL_GLContext context = SDL_GL_CreateContext(win);
    SDL_GL_MakeCurrent(win, context);

    gladLoadGLLoader(SDL_GL_GetProcAddress);
    GLuint quadsGenerated[3]{0, 0, 0};
    glCreateBuffers(2, quadsGenerated);

    const GLuint quadPositions = quadsGenerated[0];
    const GLuint quadIndices = quadsGenerated[1];

    // vec3
    constexpr float positions[] {
        -1.0f, -1.0f, 0.0f,
        +1.0f, -1.0f, 0.0f,
        -1.0f, +1.0f, 0.0f,
        +1.0f, +1.0f, 0.0f,
    };
    glNamedBufferData(quadPositions, sizeof(float) * 12, positions, GL_STATIC_DRAW);

    // ubyte
    const uint8_t indices[] = {
        0, 1, 2,
        2, 3, 1,
    };
    glNamedBufferData(quadIndices, sizeof(uint8_t) * 6, indices, GL_STATIC_DRAW);

    glCreateVertexArrays(1, &quadsGenerated[2]);
    const GLuint quad = quadsGenerated[2];

    glBindVertexArray(quad);

    glVertexArrayVertexBuffer(quad, 0, quadPositions, 0, sizeof(float) * 3);
    glVertexArrayElementBuffer(quad, quadIndices);

    glEnableVertexArrayAttrib(quad, 0);
    glVertexArrayAttribBinding(quad, 0, 0);
    glVertexArrayAttribFormat(quad, 0, 3, GL_FLOAT, GL_FALSE, 0);

    // no need to allocate any more than 1.
    GLint v;

    // vertex
    
    GLuint vert = glCreateShader(GL_VERTEX_SHADER);
    {
        std::fstream file(R"(S:\C++ Projects\rt-engine\vertex.glsl)");
        std::stringstream ss;
        ss << file.rdbuf();
        file.close();
        std::string str = ss.str();
        auto source_plain = str.c_str();
        auto source_length = str.size();
        const GLchar **source = &source_plain;
    
        glShaderSource(vert, 1, source, reinterpret_cast<GLint*>(&source_length));
        glCompileShader(vert);
    }
    
    glGetShaderiv(vert, GL_COMPILE_STATUS, &v);
    if (v == GL_FALSE) {
        char info[4096]{};
        std::memset(info, '\0', sizeof(info));
        glGetShaderInfoLog(vert, sizeof(info), nullptr, info);
        std::cout << "Vert Error: " << info << '\n';
    }

    // frag
    
    GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
    {
        std::fstream file(R"(S:\C++ Projects\rt-engine\fragment.glsl)");
        std::stringstream ss;
        ss << file.rdbuf();
        file.close();
        std::string str = ss.str();
        auto source_plain = str.c_str();
        auto source_length = str.size();
        const GLchar **source = &source_plain;
    
        glShaderSource(frag, 1, source, nullptr); // reinterpret_cast<GLint*>(&source_length));
        glCompileShader(frag);
    }

    glGetShaderiv(frag, GL_COMPILE_STATUS, &v);
    if (v == GL_FALSE) {
        char info[4096]{};
        std::memset(info, '\0', sizeof(info));
        glad_glGetShaderInfoLog(frag, sizeof(info), nullptr, info);
        std::cout << "Frag Error: " << info << '\n';
    }
    
    // program
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glValidateProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &v);
    if (v == GL_FALSE) {
        char info[4096]{};
        std::memset(info, '\0', sizeof(info));
        glGetProgramInfoLog(program, sizeof(info), nullptr, info);
        std::cout << "Program Error: " << info << '\n';
    }

    GLint uNoiseLoc = glGetUniformLocation(program, "noise");
    GLint uTemporal = glGetUniformLocation(program, "temporal");
    GLint uTemporalSampler = glGetUniformLocation(program, "temporalSampler");
    GLint uFrames = glGetUniformLocation(program, "frames");
    GLint uTime = glGetUniformLocation(program, "time");

    GLuint image;
    glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &image);
    glTextureStorage3D(image, 1, GL_RGBA8, 1920, 1080, 64);

    
    
    int frameCount = 0;
    bool quit = false;
    SDL_Event e;
    while (win != nullptr && !quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
        frameCount++;
        
        float vec[4]{};
        for (float &x : vec) {
            x = static_cast<float>(random_double());
        }

        glUseProgram(program);
        glBindVertexArray(quad);
    
        glBindImageTexture(0, image, 0, true, 0, GL_READ_WRITE, GL_RGBA8);
        glUniform1i(uTemporal, 0);
        //glBindTextureUnit(1, image);
        //glUniform1i(uTemporalSampler, 1);

        glUniform1i(uFrames, frameCount);
        glUniform4fv(uNoiseLoc, 1, vec);
        
        Uint32 ticks = SDL_GetTicks();
        f32 seconds = static_cast<f32>(static_cast<f64>(ticks) / 1000.0);
        glUniform1f(uTime, seconds * 0.150f);
        
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, nullptr);
        //glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, nullptr);

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        glFinish();
        
        SDL_GL_SwapWindow(win);
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    glDeleteTextures(1, &image);
    glDeleteShader(vert);
    glDeleteShader(frag);
    glDeleteProgram(program);
    glDeleteBuffers(2, quadsGenerated);
    glDeleteVertexArrays(1, &quadsGenerated[0]);
    
#endif
    
    SDL_Quit();
    return 0;
}
