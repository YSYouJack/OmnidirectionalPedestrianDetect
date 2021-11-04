cmake_minimum_required(VERSION 3.19.0)

project(OPD C CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Set default building type.
IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    MESSAGE(STATUS "No build type selected, default to Debug")
    SET(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (default Debug)" FORCE)
ENDIF ()

# Find OpenCV
find_package(OpenCV REQUIRED world)
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

if (MSVC)
    if ("${CMAKE_CXX_FLAGS}" MATCHES "/W[1-3]")
        string (REGEX REPLACE "/W[1-3]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else ()
        string(APPEND CMAKE_CXX_FLAGS " /W4")
    endif()

    string(APPEND CMAKE_CXX_FLAGS " /MP /wd4819 /wd4505 /wd4100")
	
	SET(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /PROFILE /INCREMENTAL:no")
	SET(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} /PROFILE /INCREMENTAL:no")

	if (MSVC_VERSION GREATER_EQUAL 1920)
		string(APPEND CMAKE_CXX_FLAGS " /wd4127 /bigobj /wd26812 /wd26451 /Zc:__cplusplus")
		set(CMAKE_CXX_STANDARD 17)
	endif()
elseif (CMAKE_COMPILER_IS_GNUCC)
	set(CMAKE_CUDA_STANDARD 11)
	add_compile_options(-Wall -Wno-unknown-pragmas -Wextra)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(HogSVM HogSVM.cpp)
target_link_libraries(HogSVM ${OpenCV_LIBS})

add_executable(MobileNet MobileNet.cpp)
target_link_libraries(MobileNet ${OpenCV_LIBS})

add_executable(ImagesToVideo ImagesToVideo.cpp)
target_link_libraries(ImagesToVideo ${OpenCV_LIBS})


if (MSVC AND NOT OpenCV_LIB_PATH MATCHES "/staticlib")
	get_target_property(__dll_dbg opencv_world IMPORTED_LOCATION_DEBUG)
	get_target_property(__dll_release opencv_world  IMPORTED_LOCATION_RELEASE)

	add_custom_command(TARGET HogSVM POST_BUILD 
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"$<$<CONFIG:debug>:${__dll_dbg}>$<$<CONFIG:release>:${__dll_release}>$<$<CONFIG:relwithdebinfo>:${__dll_release}>$<$<CONFIG:minsizerel>:${__dll_release}>"
			$<TARGET_FILE_DIR:HogSVM> 
	)
endif()