juce_add_console_app(irconv_test PRODUCT_NAME "irconv tests")

juce_generate_juce_header(irconv_test)

target_sources(irconv_test PRIVATE test_irconv.cpp)

target_compile_features(irconv_test PRIVATE cxx_std_23)

if(UNIX AND NOT APPLE)
    find_package(CURL REQUIRED)
endif()

target_link_libraries(irconv_test
    PRIVATE
        irconv
        juce::juce_recommended_warning_flags
        $<$<PLATFORM_ID:Linux>:CURL::libcurl>
)

add_test(NAME irconv_test COMMAND irconv_test)
