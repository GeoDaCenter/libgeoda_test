cmake_minimum_required(VERSION 3.2)

project(geoda_test)

###########################################################
#
# env variables
#
###########################################################

message(STATUS "Binary dir: ${CMAKE_BINARY_DIR}")

set( CMAKE_VERBOSE_MAKEFILE on )

set (DEP_LIBRARY_PATH ${CMAKE_SOURCE_DIR})
set (DEP_OS_SUB "")

if ( APPLE )
    set(CMAKE_MACOSX_RPATH 1)
    set (DEP_OS_SUB "osx")
    set (CMAKE_OSX_ARCHITECTURES "x86_64")
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -arch x86_64")

elseif (UNIX)
    set (DEP_OS_SUB "linux")
    set (CMAKE_CXX_FLAGS  "-fPIC -Wl,--no-as-needed")

elseif (MINGW)
    # i386 or x64
    set (DEP_OS_SUB "i386")

elseif (MSVC)
    # win32 or win64
    set (DEP_OS_SUB "win32")
    # if win64: 
    #set(CMAKE_GENERATOR_PLATFORM x64)
endif()

message(STATUS "DEP_OS_SUB: ${DEP_OS_SUB}")


# This little macro lets you set any XCode specific property
macro (set_xcode_property TARGET XCODE_PROPERTY XCODE_VALUE)
    set_property (TARGET ${TARGET} PROPERTY XCODE_ATTRIBUTE_${XCODE_PROPERTY} ${XCODE_VALUE})
endmacro (set_xcode_property)


###########################################################
#
#  options
#
###########################################################
option(BUILD_LIBGEODA "Builds libgeoda" ON)
if (BUILD_LIBGEODA)
    add_subdirectory(libgeoda_dylib)
endif()

option(BUILD_POSTGEODA "Builds postgeoda" OFF)
if (BUILD_POSTGEODA)
    #add_subdirectory(postgeoda)
endif()

option(BUILD_RGEODA "Builds rgeoda" OFF)

option(BUILD_PYGEODA "Builds pygeoda" OFF)
if (BUILD_PYGEODA)
    #add_subdirectory(pygeoda)
endif()

option(BUILD_JSGEODA "Builds jsgeoda" OFF)

# git clone https://github.com/google/googletest.git
option(GEODA_UNITTEST "Builds the unit test" ON)

if (GEODA_UNITTEST)
    set ( CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/Install CACHE PATH "install here" FORCE)
    add_subdirectory (googletest)
    enable_testing()
    add_subdirectory(test)
endif()

