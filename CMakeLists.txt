CMAKE_MINIMUM_REQUIRED(VERSION 3.0.1)

PROJECT(gmm)

IF (QUANTIZE_TYPE STREQUAL FIX)
    ADD_DEFINITIONS(-DFIX)
ENDIF()

FILE(GLOB UTIL_LIST_CPP ${PROJECT_SOURCE_DIR}/util/*.cpp)
MESSAGE("UTIL_LIST_CPP: " ${UTIL_LIST_CPP})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
ADD_LIBRARY(quantize SHARED ${UTIL_LIST_CPP})

SET(SOURCE ${PROJECT_SOURCE_DIR}/train.cpp)
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)

INCLUDE_DIRECTORIES("/usr/local/MATLAB/R2020b/extern/include")
LINK_DIRECTORIES("/usr/local/MATLAB/R2020b/bin/glnxa64")
#LINK_LIBRARIES("/usr/local/MATLAB/R2020b/bin/glnxa64/libmat.so" "/usr/local/MATLAB/R2020b/bin/glnxa64/libmx.so")

ADD_EXECUTABLE(gmm ${SOURCE})
TARGET_LINK_LIBRARIES(gmm quantize mat mx)
