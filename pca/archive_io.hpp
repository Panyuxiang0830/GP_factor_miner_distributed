#pragma once
#include "core.hpp"
#include "gpe/json.hpp"
#include "json_output.hpp"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace pca {
inline std::string file_fingerprint(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot fingerprint "+path);
    uint64_t h=14695981039346656037ULL;char buf[65536];
    while(in){in.read(buf,sizeof(buf));for(std::streamsize i=0;i<in.gcount();++i){h^=static_cast<unsigned char>(buf[i]);h*=1099511628211ULL;}}
    if(!in.eof())throw std::runtime_error("fingerprint read failed");
    std::ostringstream out;out<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return out.str();
}
inline int archive_integer(const gpe::Json& x) {
    if(!x.is_number()||!std::isfinite(x.number)||x.number<0||x.number>std::numeric_limits<int>::max()||std::floor(x.number)!=x.number)
        throw std::invalid_argument("invalid archive integer");
    return int(x.number);
}
inline std::vector<Record> load_archive(const std::string& path,int dimension) {
    auto j=gpe::parse_json_file(path);
    if(archive_integer(j.at("version"))!=1||archive_integer(j.at("dimension"))!=dimension||!j.at("records").is_array()||!j.at("signatures").is_string())
        throw std::invalid_argument("invalid archive schema/dimension");
    auto binary=std::filesystem::path(path).parent_path()/j.at("signatures").string;
    if(j.has("signature_hash_fnv1a64")&&file_fingerprint(binary.string())!=j.at("signature_hash_fnv1a64").string)throw std::invalid_argument("archive signature fingerprint mismatch");
    const auto& entries=j.at("records").array;
    if(std::filesystem::file_size(binary)!=entries.size()*size_t(dimension)*sizeof(float))throw std::invalid_argument("archive binary size mismatch");
    std::ifstream in(binary,std::ios::binary);std::set<std::string> keys;std::set<int> slots;std::vector<Record> records;
    for(const auto& e:entries){
        if(!e.at("key").is_string()||e.at("key").string.empty()||!keys.insert(e.at("key").string).second||!e.at("quality").is_number()||!std::isfinite(e.at("quality").number))
            throw std::invalid_argument("invalid/duplicate archive key or quality");
        int slot=archive_integer(e.at("slot"));if(size_t(slot)>=entries.size()||!slots.insert(slot).second)throw std::invalid_argument("invalid/duplicate archive slot");
        Record r{e.at("key").string,e.at("quality").number,std::vector<float>(dimension)};
        in.seekg(std::streamoff(slot)*dimension*sizeof(float));in.read(reinterpret_cast<char*>(r.signature.data()),dimension*sizeof(float));
        if(!in||!std::isfinite(similarity(r.signature,r.signature)))throw std::invalid_argument("invalid archive signature");
        records.push_back(std::move(r));
    }
    return records;
}
inline void save_archive(const std::string& path,const std::vector<Record>& records,int dimension) {
    auto binary=std::filesystem::path(path).parent_path()/"archive_signatures.f32";
    std::ofstream out(binary,std::ios::binary);gpe::Json j,entries;j.type=gpe::Json::Type::Object;entries.type=gpe::Json::Type::Array;
    auto num=[](double x){gpe::Json v;v.type=gpe::Json::Type::Number;v.number=x;return v;};
    auto str=[](const std::string& x){gpe::Json v;v.type=gpe::Json::Type::String;v.string=x;return v;};
    std::set<std::string> keys;
    for(const auto& r:records){
        if(r.key.empty()||!keys.insert(r.key).second||!std::isfinite(r.quality)||r.signature.size()!=size_t(dimension)||!std::isfinite(similarity(r.signature,r.signature)))
            throw std::invalid_argument("cannot save invalid archive record");
        gpe::Json e;e.type=gpe::Json::Type::Object;e.object["key"]=str(r.key);e.object["quality"]=num(r.quality);e.object["slot"]=num(entries.array.size());entries.array.push_back(e);
        out.write(reinterpret_cast<const char*>(r.signature.data()),dimension*sizeof(float));
    }
    out.close();if(!out)throw std::runtime_error("archive binary write failed");
    j.object["version"]=num(1);j.object["dimension"]=num(dimension);j.object["signatures"]=str(binary.filename().string());j.object["records"]=entries;
    j.object["signature_hash_fnv1a64"]=str(file_fingerprint(binary.string()));
    std::ofstream meta(path);write_json(meta,j);meta<<'\n';meta.close();if(!meta)throw std::runtime_error("archive metadata write failed");
}
}
