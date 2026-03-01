cmake_minimum_required(VERSION 3.16)
project(MapViewer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- MSVC runtime (keep your current behavior) ----
if (MSVC)
    set(CMAKE_C_FLAGS_DEBUG "/MDd")
    set(CMAKE_C_FLAGS_RELEASE "/MD")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "/MD")
endif()

# ---- GLAD ----
add_library(glad STATIC
    external/glad/src/glad.c
)

target_include_directories(glad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glad/include
)

# ---- Executable ----
add_executable(map_viewer
    src/main.cpp
    src/renderer_gl.cpp
    src/map_cache.cpp
)

target_include_directories(map_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/external/stb
)

# ---- GLFW & OpenGL ----
if (WIN32)
    set(GLFW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/libs/glfw-3.4.bin.WIN64")

    target_include_directories(map_viewer PRIVATE
        ${GLFW_ROOT}/include
    )

    target_link_libraries(map_viewer PRIVATE
        glad
        ${GLFW_ROOT}/lib-vc2022/glfw3.lib
        opengl32
    )
else()
    find_package(glfw3 REQUIRED)
    target_link_libraries(map_viewer PRIVATE glad glfw)

    if (UNIX AND NOT APPLE)
        target_link_libraries(map_viewer PRIVATE dl pthread X11)
    endif()
endif()