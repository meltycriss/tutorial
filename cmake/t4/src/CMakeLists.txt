#LINK_DIRECTORIES(/home/criss/turorial/cmake/t3/build/lib)
ADD_EXECUTABLE(main main.c)
INCLUDE_DIRECTORIES(/home/criss/turorial/cmake/t3/lib)
FIND_LIBRARY(HELLO_LIB hello HINTS /home/criss/turorial/cmake/t3/build/lib)
TARGET_LINK_LIBRARIES(main ${HELLO_LIB})
