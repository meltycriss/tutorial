import message_pb2

request=message_pb2.Request()
request.userid=1234
request.type=1
request.desc="i am a user"

request_str=request.SerializeToString()

print request_str

print "receiver:"
response=message_pb2.Request()
response.ParseFromString(request_str)
print response.userid