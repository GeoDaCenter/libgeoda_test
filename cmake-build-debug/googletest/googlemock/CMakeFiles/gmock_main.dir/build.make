# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/xunli/github/libgeoda_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/xunli/github/libgeoda_test/cmake-build-debug

# Include any dependencies generated for this target.
include googletest/googlemock/CMakeFiles/gmock_main.dir/depend.make

# Include the progress variables for this target.
include googletest/googlemock/CMakeFiles/gmock_main.dir/progress.make

# Include the compile flags for this target's objects.
include googletest/googlemock/CMakeFiles/gmock_main.dir/flags.make

googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: googletest/googlemock/CMakeFiles/gmock_main.dir/flags.make
googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o: ../googletest/googlemock/src/gmock_main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xunli/github/libgeoda_test/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o"
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gmock_main.dir/src/gmock_main.cc.o -c /Users/xunli/github/libgeoda_test/googletest/googlemock/src/gmock_main.cc

googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gmock_main.dir/src/gmock_main.cc.i"
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xunli/github/libgeoda_test/googletest/googlemock/src/gmock_main.cc > CMakeFiles/gmock_main.dir/src/gmock_main.cc.i

googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gmock_main.dir/src/gmock_main.cc.s"
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xunli/github/libgeoda_test/googletest/googlemock/src/gmock_main.cc -o CMakeFiles/gmock_main.dir/src/gmock_main.cc.s

# Object files for target gmock_main
gmock_main_OBJECTS = \
"CMakeFiles/gmock_main.dir/src/gmock_main.cc.o"

# External object files for target gmock_main
gmock_main_EXTERNAL_OBJECTS =

lib/libgmock_maind.a: googletest/googlemock/CMakeFiles/gmock_main.dir/src/gmock_main.cc.o
lib/libgmock_maind.a: googletest/googlemock/CMakeFiles/gmock_main.dir/build.make
lib/libgmock_maind.a: googletest/googlemock/CMakeFiles/gmock_main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/xunli/github/libgeoda_test/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library ../../lib/libgmock_maind.a"
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock && $(CMAKE_COMMAND) -P CMakeFiles/gmock_main.dir/cmake_clean_target.cmake
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gmock_main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
googletest/googlemock/CMakeFiles/gmock_main.dir/build: lib/libgmock_maind.a

.PHONY : googletest/googlemock/CMakeFiles/gmock_main.dir/build

googletest/googlemock/CMakeFiles/gmock_main.dir/clean:
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock && $(CMAKE_COMMAND) -P CMakeFiles/gmock_main.dir/cmake_clean.cmake
.PHONY : googletest/googlemock/CMakeFiles/gmock_main.dir/clean

googletest/googlemock/CMakeFiles/gmock_main.dir/depend:
	cd /Users/xunli/github/libgeoda_test/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/xunli/github/libgeoda_test /Users/xunli/github/libgeoda_test/googletest/googlemock /Users/xunli/github/libgeoda_test/cmake-build-debug /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock /Users/xunli/github/libgeoda_test/cmake-build-debug/googletest/googlemock/CMakeFiles/gmock_main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : googletest/googlemock/CMakeFiles/gmock_main.dir/depend

