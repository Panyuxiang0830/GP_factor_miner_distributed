#pragma once
#include "gpe/json.hpp"
#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>
namespace pca {
inline void quote_json(std::ostream& out, const std::string& text) {
    out << '"';
    for (unsigned char c : text) {
        if (c == '"') out << "\\\"";
        else if (c == '\\') out << "\\\\";
        else if (c < 32) {
            const char hex[] = "0123456789abcdef";
            out << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else out << c;
    }
    out << '"';
}
inline void write_json(std::ostream& out, const gpe::Json& j) {
    using Type = gpe::Json::Type;
    switch (j.type) {
    case Type::Null: out << "null"; break;
    case Type::Bool: out << (j.boolean ? "true" : "false"); break;
    case Type::Number:
        if (std::isfinite(j.number)) out << std::setprecision(17) << j.number;
        else out << "null";
        break;
    case Type::String: quote_json(out, j.string); break;
    case Type::Array: {
        out << '['; bool first = true;
        for (const auto& value : j.array) { if (!first) out << ','; first = false; write_json(out, value); }
        out << ']'; break;
    }
    case Type::Object: {
        out << '{'; bool first = true;
        for (const auto& item : j.object) {
            if (!first) out << ','; first = false;
            quote_json(out, item.first); out << ':'; write_json(out, item.second);
        }
        out << '}'; break;
    }
    }
}
}  // namespace pca
