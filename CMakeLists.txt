cmake_minimum_required(VERSION 3.20)
project(VIRS)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Define include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/args
    ${CMAKE_SOURCE_DIR}/third_party/embree/include/embree4
    ${CMAKE_SOURCE_DIR}/third_party/libpng
    ${CMAKE_SOURCE_DIR}/third_party/tinyobjloader
    ${CMAKE_SOURCE_DIR}/third_party/visage/build/include
    ${CMAKE_SOURCE_DIR}/third_party/visage
)

# Recursively find all .cpp files
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Set tinyobjloader cc file
set(TINYOBJLOADER_SRC "${CMAKE_SOURCE_DIR}/third_party/tinyobjloader/tiny_obj_loader.cc")


set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:/Libraries/embree/release/lib/cmake/embree-4.4.0")
find_package(embree REQUIRED)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:/Libraries/oneTBB/lib/cmake/TBB")
find_package(TBB REQUIRED CONFIG)


# Static library for PNG
set(PNG_STATIC "C:/Libraries/libpng/lib/libpng16_staticd.lib")
set(ZLIB_STATIC "${CMAKE_SOURCE_DIR}/third_party/zlib/build/Debug/zsd.lib")
set(VISAGE_LIBRARIES "${CMAKE_SOURCE_DIR}/third_party/visage/build/Debug/visage.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/_deps/bgfx-build/cmake/bgfx/Debug/bgfx.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/_deps\\bgfx-build\\cmake\\bimg\\Debug\\bimg.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/_deps\\bgfx-build\\cmake\\bimg\\Debug\\bimg_encode.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/_deps\\bgfx-build\\cmake\\bimg\\Debug\\bimg_decode.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/_deps\\bgfx-build\\cmake\\bx\\Debug\\bx.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/_deps\\freetype-build\\Debug\\freetyped.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/visage_graphics/Debug/VisageEmbeddedShaders.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/visage_graphics/Debug/VisageEmbeddedFonts.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/visage_graphics/Debug/VisageEmbeddedIcons.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/visage_graphics\\third_party\\nanosvg\\Debug\\nanosvg.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};${CMAKE_SOURCE_DIR}/third_party/visage/build/visage_graphics\\third_party\\nanosvg\\Debug\\nanosvgrast.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};d2d1.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};dwrite.lib")
set(VISAGE_LIBRARIES "${VISAGE_LIBRARIES};dxgi.lib")



# Build the executable
add_executable(VIRS ${SRC_FILES} ${TINYOBJLOADER_SRC} )
# add_library(libpng STATIC ${PNG_STATIC})
# Link against the required libraries
target_link_libraries(VIRS embree TBB::tbb ${PNG_STATIC} ${ZLIB_STATIC} ${VISAGE_LIBRARIES})

set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/build/")
set(DLL_PATHS
    "C:/Libraries/oneTBB/bin/tbb12.dll"
    "C:/Libraries/embree/release/bin/embree4.dll"
)


foreach(DLL ${DLL_PATHS})
    add_custom_command(TARGET VIRS POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${DLL}"
        "${OUTPUT_DIR}/$<CONFIGURATION>"
    )
endforeach()

# Optional: show all commands during build
set(CMAKE_VERBOSE_MAKEFILE ON)