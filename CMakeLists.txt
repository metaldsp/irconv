# Mandatory line, sets the minimum version of CMake that should be used with this repository.
cmake_minimum_required(VERSION 3.28)

# Project name and version
project(irconv VERSION 0.1.0 LANGUAGES CXX)

# Always use the newest C++ standard on green-field projects if possible.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# JUCE — fetch only when building this repo standalone.
# When included as a submodule, the parent project owns JUCE.
# ---------------------------------------------------------------------------
if(NOT TARGET juce::juce_core)
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.12
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

# Add third-party libraries.
add_subdirectory(3rdparty/FFTConvolver)

# ---------------------------------------------------------------------------
# Register the JUCE module.
#
# juce_add_module() derives the module ID from the directory name, so this
# repo must be cloned / submoduled into a folder named "irconv":
#
#   git submodule add <url> modules/irconv
#
# In the consuming project's CMakeLists.txt:
#
#   juce_add_module(${CMAKE_CURRENT_SOURCE_DIR}/modules/irconv)
#   target_link_libraries(MyTarget PRIVATE irconv)
# ---------------------------------------------------------------------------
juce_add_module("${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(irconv INTERFACE FFTConvolver)

# Tests — only when irconv is the top-level project, not when used as a submodule
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
