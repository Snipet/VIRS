cmake_minimum_required(VERSION 3.20)
project(VIRS LANGUAGES CXX)

option(VIRS_WITH_CUDA "Build GPU kernels" ON)      # toggle

if (VIRS_WITH_CUDA)
    add_compile_definitions(VIRS_WITH_CUDA=1)
endif()


if (VIRS_WITH_CUDA)
    enable_language(CUDA)      # Conditionally enable CUDA
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SRC_CPP CONFIGURE_DEPENDS src/*.cpp)
set(TINYOBJ_SRC third_party/tinyobjloader/tiny_obj_loader.cc)

if (VIRS_WITH_CUDA)
    file(GLOB_RECURSE SRC_CU CONFIGURE_DEPENDS src/*.cu)
endif()

add_executable(VIRS ${SRC_CPP} ${TINYOBJ_SRC} ${SRC_CU})


target_include_directories(VIRS PRIVATE
    include
    src
    third_party/args
    third_party/embree/include
    third_party/tinyobjloader
    third_party/libpng
)

target_compile_options(VIRS PRIVATE
    -Wall -Wextra -Wshadow -Wpedantic
    $<$<CONFIG:Release>:
        -O3 -march=native -mavx512f -ffast-math -funroll-loops -fopenmp-simd
    >
    $<$<CONFIG:Debug>:
        -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer
    >
)


if (VIRS_WITH_CUDA)
    # Pick your target architecture(s). 86 = Ampere A10/A40; 80 = A100; 75 = T4
    set(CMAKE_CUDA_ARCHITECTURES 86)

    target_compile_options(VIRS PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math -lineinfo -Xcompiler=-fopenmp-simd
        >
    )

    # NVCC treats .cu by default; ensure -std=c++17 same as host
    set_target_properties(VIRS PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17)

    # Link against cuda runtime explicitly (optional, but clearer)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(VIRS PRIVATE CUDA::cudart)
endif()

file(GLOB_RECURSE EMBREE_LIBS third_party/embree/build/*.a)

find_package(TBB       REQUIRED)
find_package(ZLIB      REQUIRED)
#find_package(benchmark REQUIRED)
find_package(PNG       REQUIRED)

target_link_libraries(VIRS PRIVATE
    ${EMBREE_LIBS}
    TBB::tbb
    ZLIB::ZLIB
    PNG::PNG
#   benchmark::benchmark
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE        ON)
