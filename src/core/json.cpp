#include "gpe/json.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpe {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text) {}

    Json parse_document() {
        Json value = parse_value();
        skip_ws();
        if (pos_ != s_.size()) fail("trailing characters");
        return value;
    }

private:
    const std::string& s_;
    size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(pos_) + ": " + what);
    }

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\t' || s_[pos_] == '\r'))
            ++pos_;
    }

    char peek() {
        skip_ws();
        if (pos_ >= s_.size()) fail("unexpected end of input");
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        ++pos_;
    }

    bool consume_literal(const char* word) {
        size_t n = std::char_traits<char>::length(word);
        if (s_.compare(pos_, n, word) == 0) {
            pos_ += n;
            return true;
        }
        return false;
    }

    Json parse_value() {
        char c = peek();
        if (c == '[') return parse_array();
        if (c == '{') return parse_object();
        if (c == '"') {
            Json v;
            v.type = Json::Type::String;
            v.string = parse_string();
            return v;
        }
        Json v;
        if (consume_literal("null")) return v;
        if (consume_literal("true")) { v.type = Json::Type::Bool; v.boolean = true; return v; }
        if (consume_literal("false")) { v.type = Json::Type::Bool; return v; }
        if (consume_literal("NaN")) { v.type = Json::Type::Number; v.number = NAN; return v; }
        return parse_number();
    }

    Json parse_number() {
        const char* begin = s_.c_str() + pos_;
        char* end = nullptr;
        double x = std::strtod(begin, &end);
        if (end == begin) fail("invalid value");
        pos_ += static_cast<size_t>(end - begin);
        Json v;
        v.type = Json::Type::Number;
        v.number = x;
        return v;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) fail("unterminated string");
            char c = s_[pos_++];
            if (c == '"') break;
            if (c != '\\') { out.push_back(c); continue; }
            if (pos_ >= s_.size()) fail("bad escape");
            char e = s_[pos_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (pos_ + 4 > s_.size()) fail("bad unicode escape");
                    unsigned code = std::stoul(s_.substr(pos_, 4), nullptr, 16);
                    pos_ += 4;
                    // Encode the BMP code point as UTF-8 (surrogate pairs are not needed here).
                    if (code < 0x80) {
                        out.push_back(static_cast<char>(code));
                    } else if (code < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }

    Json parse_array() {
        expect('[');
        Json v;
        v.type = Json::Type::Array;
        if (peek() == ']') { ++pos_; return v; }
        while (true) {
            v.array.push_back(parse_value());
            char c = peek();
            ++pos_;
            if (c == ']') break;
            if (c != ',') fail("expected ',' or ']'");
        }
        return v;
    }

    Json parse_object() {
        expect('{');
        Json v;
        v.type = Json::Type::Object;
        if (peek() == '}') { ++pos_; return v; }
        while (true) {
            if (peek() != '"') fail("expected object key");
            std::string key = parse_string();
            expect(':');
            v.object[key] = parse_value();
            char c = peek();
            ++pos_;
            if (c == '}') break;
            if (c != ',') fail("expected ',' or '}'");
        }
        return v;
    }
};

void dump_to(const Json& v, std::ostringstream& os) {
    switch (v.type) {
        case Json::Type::Null: os << "null"; break;
        case Json::Type::Bool: os << (v.boolean ? "true" : "false"); break;
        case Json::Type::Number:
            if (v.number == std::floor(v.number) && std::fabs(v.number) < 1e15)
                os << static_cast<long long>(v.number);
            else
                os << v.number;
            break;
        case Json::Type::String: os << '"' << v.string << '"'; break;
        case Json::Type::Array:
            os << '[';
            for (size_t i = 0; i < v.array.size(); ++i) {
                if (i) os << ',';
                dump_to(v.array[i], os);
            }
            os << ']';
            break;
        case Json::Type::Object: {
            os << '{';
            bool first = true;
            for (const auto& [key, value] : v.object) {
                if (!first) os << ',';
                first = false;
                os << '"' << key << "\":";
                dump_to(value, os);
            }
            os << '}';
            break;
        }
    }
}

}  // namespace

std::string Json::dump() const {
    std::ostringstream os;
    dump_to(*this, os);
    return os.str();
}

Json parse_json(const std::string& text) { return Parser(text).parse_document(); }

Json parse_json_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return parse_json(buffer.str());
}

}  // namespace gpe
