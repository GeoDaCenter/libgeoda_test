
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################
# Add test cpp file
set (TEST_SOURCE_FILES
        ./test_weights.cpp
        ./test_localSA.cpp
        ./test_clustering.cpp
        ./test_algorithms.cpp)

add_executable( runUnitTests ${TEST_SOURCE_FILES} )

set_target_properties(runUnitTests PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO)

#find_package(Boost 1.57.0 COMPONENTS date_time thread system)
#if(Boost_FOUND)
#    target_include_directories(runUnitTests PUBLIC ${Boost_INCLUDE_DIR} )
#    target_link_libraries(runUnitTests ${Boost_LIBRARIES})
#endif()
#find_package(GDAL)
#if (GDAL_FOUND)
#    message(STATUS "GDAL header: ${GDAL_INCLUDE_DIR}")
#    message(STATUS "GDAL libs: ${GDAL_LIBRARY}")
#    target_include_directories(runUnitTests PUBLIC ${GDAL_INCLUDE_DIR} )
#    target_link_libraries(runUnitTests ${GDAL_LIBRARY})
#endif()

target_include_directories(runUnitTests PUBLIC
        ${gtest_SOURCE_DIR}/include
        ${gmock_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/../boost_static/include
        ${DEP_LIBRARY_PATH}/libgeoda_static/include
        )
# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests gtest gtest_main geoda
        ${CMAKE_SOURCE_DIR}/../boost_static/lib/${DEP_OS_SUB}/libboost_date_time.a
        ${CMAKE_SOURCE_DIR}/../boost_static/lib/${DEP_OS_SUB}/libboost_system.a
        ${CMAKE_SOURCE_DIR}/../boost_static/lib/${DEP_OS_SUB}/libboost_thread.a
        )

add_test( runUnitTests runUnitTests )
