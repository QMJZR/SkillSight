#!/usr/bin/env python3
"""下载 SkillRouter 和 SRA-Bench 数据集到当前目录。

依赖：
    pip install -U huggingface_hub

使用：
    python download_datasets.py
"""

from pathlib import Path
from huggingface_hub import snapshot_download


def download_skillrouter(target_dir: Path):
    """下载 SkillRouter-Eval-Core 数据集。

    Args:
        target_dir: 目标目录路径
    """
    print("\n" + "="*60)
    print("开始下载 SkillRouter-Eval-Core 数据集...")
    print("="*60)
    print(f"目标路径: {target_dir}")
    print("数据集: pipizhao/SkillRouter-Eval-Core")
    print("预计大小: ~422 MB")
    print()

    snapshot_download(
        repo_id="pipizhao/SkillRouter-Eval-Core",
        repo_type="dataset",
        local_dir=target_dir,
        allow_patterns=[
            "manifest.json",
            "tasks.jsonl",
            "relevance.json",
            "easy/*.jsonl.gz",
            "hard/*.jsonl.gz",
        ],
    )

    print(f"\n✓ SkillRouter 数据集下载完成: {target_dir}")
    print(f"  - 87 个路由任务")
    print(f"  - Easy 候选池: 78,361 条技能")
    print(f"  - Hard 候选池: 79,141 条技能")


def download_sra_bench(target_dir: Path):
    """下载 SRA-Bench 数据集。

    Args:
        target_dir: 目标目录路径
    """
    print("\n" + "="*60)
    print("开始下载 SRA-Bench 数据集...")
    print("="*60)
    print(f"目标路径: {target_dir}")
    print("数据集: WeihangSu/SRA-Bench")
    print("预计大小: ~242 MB")
    print()

    snapshot_download(
        repo_id="WeihangSu/SRA-Bench",
        repo_type="dataset",
        local_dir=target_dir,
    )

    print(f"\n✓ SRA-Bench 数据集下载完成: {target_dir}")
    print(f"  - 5,400 个测试实例")
    print(f"  - 636 个 gold skills")
    print(f"  - 26,262 个完整技能语料库")


def verify_downloads(skillrouter_dir: Path, sra_bench_dir: Path):
    """验证下载的文件是否完整。

    Args:
        skillrouter_dir: SkillRouter 数据集目录
        sra_bench_dir: SRA-Bench 数据集目录
    """
    print("\n" + "="*60)
    print("验证下载结果...")
    print("="*60)

    # 验证 SkillRouter
    skillrouter_files = [
        "manifest.json",
        "tasks.jsonl",
        "relevance.json",
        "easy",
        "hard",
    ]

    print(f"\n[SkillRouter] 检查路径: {skillrouter_dir}")
    all_ok = True
    for item in skillrouter_files:
        path = skillrouter_dir / item
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {item}")
        if not exists:
            all_ok = False

    # 验证 SRA-Bench
    sra_bench_files = [
        "corpus/corpus.json",
        "instances/theoremqa.json",
        "instances/logicbench.json",
        "instances/toolqa.json",
        "instances/medcalcbench.json",
        "instances/champ.json",
        "instances/bigcodebench.json",
    ]

    print(f"\n[SRA-Bench] 检查路径: {sra_bench_dir}")
    for item in sra_bench_files:
        path = sra_bench_dir / item
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {item}")
        if not exists:
            all_ok = False

    print()
    if all_ok:
        print("✓ 所有文件验证通过！")
    else:
        print("✗ 部分文件缺失，请检查下载过程是否有错误。")

    return all_ok


def main():
    """主函数：下载两个数据集到当前目录。"""
    # 设置下载目录
    root = Path(".")
    skillrouter_dir = root / "SkillRouter-Eval-Core"
    sra_bench_dir = root / "SRA-Bench"

    print("="*60)
    print("数据集下载脚本")
    print("="*60)
    print(f"工作目录: {root.resolve()}")
    print(f"SkillRouter 目标: {skillrouter_dir}")
    print(f"SRA-Bench 目标: {sra_bench_dir}")
    print(f"预计总下载大小: ~664 MB")
    print()

    try:
        # 创建目录
        skillrouter_dir.mkdir(parents=True, exist_ok=True)
        sra_bench_dir.mkdir(parents=True, exist_ok=True)

        # 下载数据集
        download_skillrouter(skillrouter_dir)
        download_sra_bench(sra_bench_dir)

        # 验证下载
        verify_downloads(skillrouter_dir, sra_bench_dir)

        # 输出最终结构
        print("\n" + "="*60)
        print("下载完成！最终目录结构：")
        print("="*60)
        print(f"""
当前目录/
├── SkillRouter-Eval-Core/
│   ├── easy/
│   │   ├── part-00000.jsonl.gz
│   │   └── ... (共 10 个分片)
│   ├── hard/
│   │   ├── part-00000.jsonl.gz
│   │   └── ... (共 10 个分片)
│   ├── manifest.json
│   ├── relevance.json
│   └── tasks.jsonl
└── SRA-Bench/
    ├── corpus/
    │   └── corpus.json (~232 MB)
    └── instances/
        ├── theoremqa.json
        ├── logicbench.json
        ├── toolqa.json
        ├── medcalcbench.json
        ├── champ.json
        └── bigcodebench.json
        """)

        print("\n提示：")
        print("  - SkillRouter 的 .jsonl.gz 文件无需手动解压")
        print("  - SRA-Bench 的 corpus.json 已经是解压后的 JSON 文件")
        print("  - 两个数据集可直接用于复现实验")

    except Exception as e:
        print(f"\n✗ 下载过程中出现错误: {e}")
        print("\n请检查：")
        print("  1. 是否已安装 huggingface_hub: pip install -U huggingface_hub")
        print("  2. 网络连接是否正常")
        print("  3. 磁盘空间是否充足（建议至少 1 GB）")
        raise


if __name__ == "__main__":
    main()
