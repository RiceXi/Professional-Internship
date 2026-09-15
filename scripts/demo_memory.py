"""Deterministic, CPU-only course demo of sharing, COW, swap and reclaim."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.memory import PagedMemory


def main():
    mem = PagedMemory(2, 4, host_pages=4)
    parent = mem.new_sequence()
    for token in [10, 20, 30]:
        mem.append(parent, token)
    child = mem.fork(parent)
    print('1. fork 后父子共享一页：', mem.metrics())
    mem.append(child, 99)
    assert mem.read_sequence(parent) == [10, 20, 30]
    assert mem.read_sequence(child) == [10, 20, 30, 99]
    print('2. 子分支写入触发 COW；父序列：', mem.read_sequence(parent))
    print('   子序列：', mem.read_sequence(child))
    others = []
    for token in [40, 50, 60]:
        sid = mem.new_sequence()
        mem.append(sid, token)
        others.append(sid)
    print('3. 两个物理页承载五个独立页，LRU 换出：', mem.metrics())
    assert mem.read_sequence(child) == [10, 20, 30, 99]
    print('4. 重新访问子分支，内容完整换回：', mem.read_sequence(child))
    with mem.pin_sequences([parent]) as table:
        print('5. 前向期间固定父页，当前物理块表：', table)
        mem.check_invariants()
    for sid in [parent, child, *others]:
        mem.free(sid)
    mem.check_invariants()
    assert len(mem.free_frames) == 2 and len(mem.free_host) == 4
    print('6. 全部回收，无页面泄漏：', mem.metrics())
    print('以上为 CPU 页内容演示；真实 CUDA/Qwen3 验证请运行 scripts/validate_gpu.sh。')


if __name__ == '__main__':
    main()
