"""
可视化和解释 SAE 特征
"""
import json
import argparse
from collections import Counter
import re


def extract_spatial_relations(text: str) -> list:
    """从文本中提取空间关系词"""
    relations = []
    patterns = [
        r'\b(left|right|above|below|front|behind)\b',
        r'\b(is\s+\w+\s+of)\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        relations.extend(matches)
    
    return relations


def analyze_feature_patterns(examples: list) -> dict:
    """分析特征激活的模式"""
    # 提取所有文本
    texts = [ex["prompt"] for ex in examples]
    
    # 统计空间关系词
    all_relations = []
    for text in texts:
        all_relations.extend(extract_spatial_relations(text))
    
    relation_counts = Counter(all_relations)
    
    # 统计句子长度
    sentence_counts = [len(text.split('.')) for text in texts]
    avg_sentences = sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0
    
    # 统计实体数量（通过大写字母判断）
    entity_counts = [len(re.findall(r'\b[A-Z]\b', text)) for text in texts]
    avg_entities = sum(entity_counts) / len(entity_counts) if entity_counts else 0
    
    return {
        "common_relations": dict(relation_counts.most_common(10)),
        "avg_sentences": avg_sentences,
        "avg_entities": avg_entities,
        "num_examples": len(examples),
    }


def visualize_feature(feature_file: str, output_txt: str = None):
    """可视化单个特征"""
    with open(feature_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    feature_id = data["feature_id"]
    examples = data["top_examples"]
    
    print(f"\n{'='*80}")
    print(f"Feature {feature_id} Analysis")
    print(f"{'='*80}")
    
    # 分析模式
    patterns = analyze_feature_patterns(examples)
    
    print(f"\n📊 Statistics:")
    print(f"  - Number of top examples: {patterns['num_examples']}")
    print(f"  - Average sentences per example: {patterns['avg_sentences']:.1f}")
    print(f"  - Average entities per example: {patterns['avg_entities']:.1f}")
    
    print(f"\n🔤 Common spatial relations:")
    for relation, count in patterns['common_relations'].items():
        print(f"  - {relation}: {count} times")
    
    print(f"\n📝 Top 10 examples:")
    for i, ex in enumerate(examples[:10], 1):
        score = ex["score"]
        prompt = ex["prompt"].replace("\n", " ").strip()
        if len(prompt) > 150:
            prompt = prompt[:147] + "..."
        print(f"\n  {i}. Score: {score:.4f}")
        print(f"     {prompt}")
    
    # 保存到文本文件
    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(f"Feature {feature_id} Analysis\n")
            f.write("="*80 + "\n\n")
            f.write("Statistics:\n")
            f.write(f"  - Number of top examples: {patterns['num_examples']}\n")
            f.write(f"  - Average sentences: {patterns['avg_sentences']:.1f}\n")
            f.write(f"  - Average entities: {patterns['avg_entities']:.1f}\n\n")
            f.write("Common relations:\n")
            for relation, count in patterns['common_relations'].items():
                f.write(f"  - {relation}: {count}\n")
            f.write("\nTop examples:\n")
            for i, ex in enumerate(examples[:20], 1):
                f.write(f"\n{i}. Score: {ex['score']:.4f}\n")
                f.write(f"   {ex['prompt']}\n")
        print(f"\n✅ Saved detailed report to: {output_txt}")


def compare_features(stats_file: str, top_n: int = 20):
    """比较多个特征"""
    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"Feature Comparison")
    print(f"{'='*80}")
    
    # 按激活频率排序
    sorted_by_freq = sorted(
        stats.items(),
        key=lambda x: x[1]["activation_frequency"],
        reverse=True
    )
    
    print(f"\n🔥 Top {top_n} most frequently activated features:")
    print(f"{'Feature':<10} {'Frequency':<15} {'Mean Act':<15} {'Max Act':<15}")
    print("-" * 60)
    for feat_id, stat in sorted_by_freq[:top_n]:
        print(f"{feat_id:<10} {stat['activation_frequency']:<15.4f} "
              f"{stat['mean_activation']:<15.6f} {stat['max_activation']:<15.4f}")
    
    # 按平均激活值排序
    sorted_by_mean = sorted(
        stats.items(),
        key=lambda x: x[1]["mean_activation"],
        reverse=True
    )
    
    print(f"\n⚡ Top {top_n} features with highest mean activation:")
    print(f"{'Feature':<10} {'Mean Act':<15} {'Frequency':<15} {'Max Act':<15}")
    print("-" * 60)
    for feat_id, stat in sorted_by_mean[:top_n]:
        print(f"{feat_id:<10} {stat['mean_activation']:<15.6f} "
              f"{stat['activation_frequency']:<15.4f} {stat['max_activation']:<15.4f}")
    
    # 死特征（从不激活）
    dead_features = [
        feat_id for feat_id, stat in stats.items()
        if stat["activation_frequency"] == 0
    ]
    
    print(f"\n💀 Dead features (never activated): {len(dead_features)} / {len(stats)}")
    if len(dead_features) > 0 and len(dead_features) <= 50:
        print(f"   {dead_features}")
    
    # 稀疏度分析
    active_features = [
        stat["activation_frequency"] for stat in stats.values()
        if stat["activation_frequency"] > 0
    ]
    if active_features:
        avg_freq = sum(active_features) / len(active_features)
        print(f"\n📈 Sparsity statistics:")
        print(f"   - Average activation frequency (active features): {avg_freq:.4f}")
        print(f"   - Percentage of active features: {len(active_features)/len(stats)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="可视化 SAE 特征")
    parser.add_argument("--stats", type=str, help="所有特征的统计文件 (from --analyze_all)")
    parser.add_argument("--feature", type=str, help="单个特征的例子文件")
    parser.add_argument("--output", type=str, help="输出文本文件路径")
    parser.add_argument("--top_n", type=int, default=20, help="显示 top N 个特征")
    
    args = parser.parse_args()
    
    if args.stats:
        compare_features(args.stats, top_n=args.top_n)
    
    if args.feature:
        visualize_feature(args.feature, output_txt=args.output)
    
    if not args.stats and not args.feature:
        parser.print_help()


if __name__ == "__main__":
    main()



