add_library(FFTConvolver STATIC
    AudioFFT.cpp
    FFTConvolver.cpp
    TwoStageFFTConvolver.cpp
    Utilities.cpp
)

target_include_directories(FFTConvolver SYSTEM
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Suppress warnings for third-party code
if(MSVC)
    target_compile_options(FFTConvolver PRIVATE /W0)
else()
    target_compile_options(FFTConvolver PRIVATE -w)
endif()
