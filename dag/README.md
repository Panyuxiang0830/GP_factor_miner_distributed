# dag/ — DAG planner tools (part ②)

Code for the DAG between Python expression generation and the C++ engine:
node statistics, memory scheduling, cross-batch cache simulation, S4
reproduction and the S5 plan contract. The numerical engine is not changed.

- Usage and results: [docs/dag/README.md](../docs/dag/README.md)
- Task description: [docs/dag/task.md](../docs/dag/task.md)
- S5 interface: [docs/dag/S5_interface_zh.txt](../docs/dag/S5_interface_zh.txt)

Run the tests from the repository root:

```bash
python3 -m unittest discover -s dag -p 'test_*.py' -v
```
