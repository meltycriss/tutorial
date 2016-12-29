#include "./scene.pb.h"
#include <iostream>
#include <fstream>
using namespace std;

int main()

{

  using namespace scene;

  Uavs uavs;
  Uav *uav;
  uav = uavs.add_uav();
  uav->set_x(-2.);
  uav->set_y( 2.);
  uav = uavs.add_uav();
  uav->set_x(-2.);
  uav->set_y( 2.);
  uav = uavs.add_uav();
  uav->set_x( 2.);
  uav->set_y( 2.);
  uav = uavs.add_uav();
  uav->set_x( 2.);
  uav->set_y( 2.);

  string str;

  uavs.SerializeToString(&str); // 将对象序列化到字符串，除此外还可以序列化到fstream等

  fstream fd("data.txt", ios::out | ios::in);
  for(int i=0; i<2; ++i){
    fd << str << endl << "###" << endl;
  }
  fd.close();

  printf("%s\n", str.c_str());

  Uavs readUavs;

  readUavs.ParseFromString(str); // 从字符串反序列化
  for(int i=0; i<readUavs.uav_size(); ++i){
    Uav readUav = readUavs.uav(i);
    cout << readUav.x() << " " << readUav.y() << endl;
  }

  return 0;

}
