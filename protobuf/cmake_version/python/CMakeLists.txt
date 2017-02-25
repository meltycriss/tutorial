FILE(GLOB proto_files ${CMAKE_CURRENT_SOURCE_DIR}/../src/proto/*.proto)  
FOREACH(proto_file ${proto_files})
    STRING(REGEX REPLACE "[^/]proto" "" proto_file_name ${proto_file})
    #    LIST(APPEND PROTO_SRCS "${proto_file_name}.pb.cc")
message(STATUS ${proto_file_name})
	ADD_CUSTOM_COMMAND(
		OUTPUT "${proto_file_name}_pb2.py"
    COMMAND protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
    --python_out=${CMAKE_CURRENT_BINARY_DIR} ${proto_file} 
	)
ENDFOREACH(proto_file)
#include_directories(./proto)
#add_executable(demo main.cpp ${PROTO_SRCS}) 
#target_link_libraries(demo protobuf)
