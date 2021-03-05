#############################################################
#
# libgeoda_dylib library
#
#############################################################
project(geoda)

# Remove 'lib' prefix for shared libraries on Windows
if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

#add_subdirectory(${CMAKE_SOURCE_DIR}/libgeoda)
file(GLOB GEODA_SOURCE_FILES
        ${CMAKE_SOURCE_DIR}/libgeoda/*.cpp
        ${CMAKE_SOURCE_DIR}/libgeoda/*/*.cpp
        ${CMAKE_SOURCE_DIR}/libgeoda/*/*.c)

message(STATUS "GEODA_SOURCE_FILES: ${GEODA_SOURCE_FILES}")

add_library(${PROJECT_NAME} STATIC ${GEODA_SOURCE_FILES})

# Add install target for libgeoda_dylib
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

#  install header files
foreach ( file ${GEODA_INCLUDE_FILES} )
    get_filename_component( dir ${file} DIRECTORY )
    install( FILES ../libgeoda/${file} DESTINATION include/libgeoda/${dir} )
endforeach()

# PRIVATE - use only for this specified build target
# PUBLIC - use it for specified target and for targets which links with this project
# INTERFACE -- use it only for targets which links with the current project

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_SOURCE_DIR}/boost
        )

set(USE_BOOST FALSE)

if (USE_BOOST)
    message(STATUS "using boost libs")
    # Link test executable against gtest & gtest_main
    target_link_libraries(${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/boost/stage/lib/libboost_date_time.a
        ${CMAKE_SOURCE_DIR}/boost/stage/lib/libboost_system.a
        ${CMAKE_SOURCE_DIR}/boost/stage/lib/libboost_thread.a
    )
else()
    message(STATUS "using pthreads")
    set(GEODA_CFLAGS "-D__USE_PTHREAD__")
    set(GEODA_CPPFLAGS "-D__USE_PTHREAD__")
    #set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GEODA_CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CPP_FLAGS} ${GEODA_CPPFLAGS}")
endif()

# Expose libgeoda_dylib's public includes (including the ones of C's dependencies transitively)
# to other subprojects through cache variable.
set(lib${PROJECT_NAME}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

