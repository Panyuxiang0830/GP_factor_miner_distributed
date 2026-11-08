// Uses the handed-off C++ DAG directly. stdout contains private expressions;
// the Python caller keeps it in memory and writes only aggregate checks.
#include "gpe/dag.hpp"
#include <iostream>
#include <stdexcept>
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("expected batch JSON path");
 auto input=gpe::parse_json_file(argv[1]);if(!input.is_array())throw std::runtime_error("expected JSON list");
 gpe::Dag dag;std::vector<int> accepted;
 for(size_t i=0;i<input.array.size();++i)if(dag.add_expression(input.array[i])>=0)accepted.push_back(i);
 dag.finalize();std::vector<std::string> expressions;
 std::cout<<"{\"nodes\":[";
 for(size_t i=0;i<dag.nodes().size();++i){const auto& n=dag.nodes()[i];std::string e;
 if(n.op==gpe::Op::Leaf){gpe::Json leaf;leaf.type=gpe::Json::Type::String;leaf.string=dag.fields()[n.field];e=leaf.dump();}
 else{e="[\""+std::string(gpe::op_info(n.op).name)+"\","+expressions[n.a];if(n.b>=0)e+=","+expressions[n.b];
 auto kind=gpe::op_info(n.op).kind;if(kind==gpe::OpKind::Rolling||kind==gpe::OpKind::PairRolling)e+=","+std::to_string(n.window);e+="]";}
 expressions.push_back(e);if(i)std::cout<<",";
 std::cout<<"{\"expression\":"<<e<<",\"level\":"<<n.level<<",\"tree_nodes\":"<<n.tree_nodes<<",\"parents\":"<<n.parents<<"}";
 }
 std::cout<<"],\"accepted_indices\":[";for(size_t i=0;i<accepted.size();++i){if(i)std::cout<<",";std::cout<<accepted[i];}
 std::cout<<"],\"roots\":[";for(size_t i=0;i<dag.roots().size();++i){if(i)std::cout<<",";std::cout<<dag.roots()[i];}std::cout<<"]}\n";
 return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
