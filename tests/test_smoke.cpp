#include <cstdio>

#include "gpe/version.hpp"

int main() {
    std::printf("%s\n", gpe::build_info().c_str());
    return gpe::build_info().empty() ? 1 : 0;
}
