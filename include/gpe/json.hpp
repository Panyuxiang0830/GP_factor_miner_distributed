#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gpe {

// Minimal JSON value: enough for expression batches, metadata and test fixtures.
struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    bool is_null() const { return type == Type::Null; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool has(const std::string& key) const { return is_object() && object.count(key) > 0; }
    const Json& at(const std::string& key) const { return object.at(key); }

    // Compact serialisation, used for diagnostics.
    std::string dump() const;
};

// Throws std::runtime_error on malformed input.
Json parse_json(const std::string& text);
Json parse_json_file(const std::string& path);

}  // namespace gpe
