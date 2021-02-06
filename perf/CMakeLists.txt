
################################
# Performance Tests
################################
# Add test cpp file
set (PERF_SOURCE_FILES
        #./perf_local_moran.cpp
        ./perf_local_joincount.cpp
        )

add_executable(perf_libgeoda ${PERF_SOURCE_FILES} )

set_target_properties(perf_libgeoda PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS YES)

target_include_directories(perf_libgeoda PUBLIC
        ${libgeoda_INCLUDE_DIRS}
        )

target_link_libraries(perf_libgeoda  geoda)

