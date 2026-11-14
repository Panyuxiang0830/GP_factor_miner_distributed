#pragma once
#include "gpe/json.hpp"
#include "json_output.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pca::bench {
using Json=gpe::Json;
using Clock=std::chrono::steady_clock;
inline double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
inline Json object() { Json j;j.type=Json::Type::Object;return j; }
inline Json array() { Json j;j.type=Json::Type::Array;return j; }
inline Json number(double x) { Json j;if(std::isfinite(x)){j.type=Json::Type::Number;j.number=x;}return j; }
inline Json string(const std::string& x) { Json j;j.type=Json::Type::String;j.string=x;return j; }
inline void put(Json& j,const std::string& k,double x) { j.object[k]=number(x); }
inline void put(Json& j,const std::string& k,const std::string& x) { j.object[k]=string(x); }
inline void write(const std::filesystem::path& p,const Json& j) {
    std::ofstream f(p);write_json(f,j);f<<'\n';if(!f)throw std::runtime_error("cannot write "+p.string());
}
inline std::map<std::string,std::string> options(int argc,char** argv) {
    std::map<std::string,std::string> x;
    for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::invalid_argument("options need values");x[argv[i]]=argv[i+1];}
    return x;
}
template<class T> std::vector<T> read(const std::string& p,size_t offset,size_t count) {
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("cannot open "+p);
    f.seekg(std::streamoff(offset)*sizeof(T));std::vector<T> x(count);
    f.read(reinterpret_cast<char*>(x.data()),std::streamsize(count)*sizeof(T));
    if(!f)throw std::runtime_error("short read "+p);return x;
}
template<class T> void binary(const std::filesystem::path& p,const std::vector<T>& x) {
    std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(x.data()),std::streamsize(x.size())*sizeof(T));
    if(!f)throw std::runtime_error("cannot write "+p.string());
}
inline double cosine(const std::vector<float>& a,const float* b) {
    double xy=0,xx=0,yy=0;for(size_t i=0;i<a.size();++i){xy+=double(a[i])*b[i];xx+=double(a[i])*a[i];yy+=double(b[i])*b[i];}
    if(!(xx>0&&yy>0))throw std::runtime_error("invalid comparison signature");
    return std::min(1.,std::fabs(xy)/(std::sqrt(xx)*std::sqrt(yy)));
}
}
