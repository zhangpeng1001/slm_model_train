"""
数据集质量分析脚本
分析5个数据集的格式、内容质量和优化建议
"""

import json
import os
from collections import Counter
import re

# 数据集配置
DATASETS = {
    "data_extraction": {
        "path": r"E:\project\python\slm_model_train\src\dataset\data_platform\data_extraction.json",
        "name": "数据提取数据集"
    },
    "dp_qa": {
        "path": r"E:\project\python\slm_model_train\src\dataset\data_platform\dp_qa.json",
        "name": "数据平台问答数据集"
    },
    "question_classifier": {
        "path": r"E:\project\python\slm_model_train\src\dataset\data_platform\question_classifier.json",
        "name": "问题分类数据集"
    },
    "question_type_classifier": {
        "path": r"E:\project\python\slm_model_train\src\dataset\data_platform\question_type_classifier.json",
        "name": "问题类型分类数据集"
    },
    "tool_data_platform": {
        "path": r"E:\project\python\slm_model_train\src\dataset\data_platform\tool_data_platform.json",
        "name": "工具调用数据集"
    }
}

def load_dataset(file_path):
    """加载数据集"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载失败: {e}")
        return None

def analyze_basic_stats(data, dataset_name):
    """分析基础统计信息"""
    print(f"\n=== {dataset_name} 基础统计 ===")
    print(f"样本总数: {len(data)}")
    
    # 分析输入长度
    input_lengths = [len(item.get('input', '')) for item in data]
    output_lengths = [len(item.get('output', '')) for item in data]
    
    print(f"输入长度统计:")
    print(f"  平均长度: {sum(input_lengths)/len(input_lengths):.1f}")
    print(f"  最短: {min(input_lengths)}, 最长: {max(input_lengths)}")
    
    print(f"输出长度统计:")
    print(f"  平均长度: {sum(output_lengths)/len(output_lengths):.1f}")
    print(f"  最短: {min(output_lengths)}, 最长: {max(output_lengths)}")

def analyze_data_extraction(data):
    """分析数据提取数据集"""
    print(f"\n=== 数据提取数据集 专项分析 ===")
    
    # 分析提取的数据类型
    outputs = [item['output'] for item in data]
    output_counter = Counter(outputs)
    
    print(f"提取的数据类型分布:")
    for output, count in output_counter.most_common(10):
        print(f"  {output}: {count}次")
    
    # 检查数据质量
    print(f"\n质量检查:")
    unique_inputs = len(set(item['input'] for item in data))
    print(f"  输入去重后: {unique_inputs}个 (原始: {len(data)}个)")
    
    # 分析输入模式
    patterns = []
    for item in data:
        input_text = item['input']
        if '怎么' in input_text or '如何' in input_text:
            patterns.append('询问方式')
        elif '需要' in input_text or '想要' in input_text:
            patterns.append('需求表达')
        elif '数据' in input_text:
            patterns.append('直接描述')
        else:
            patterns.append('其他')
    
    pattern_counter = Counter(patterns)
    print(f"输入模式分布:")
    for pattern, count in pattern_counter.items():
        print(f"  {pattern}: {count}次")

def analyze_classification(data, dataset_name):
    """分析分类数据集"""
    print(f"\n=== {dataset_name} 专项分析 ===")
    
    # 分析类别分布
    outputs = [item['output'] for item in data]
    output_counter = Counter(outputs)
    
    print(f"类别分布:")
    for output, count in output_counter.items():
        print(f"  {output}: {count}次 ({count/len(data)*100:.1f}%)")
    
    # 检查类别平衡性
    min_count = min(output_counter.values())
    max_count = max(output_counter.values())
    balance_ratio = min_count / max_count
    
    print(f"类别平衡性:")
    print(f"  最少类别样本数: {min_count}")
    print(f"  最多类别样本数: {max_count}")
    print(f"  平衡比例: {balance_ratio:.2f} (1.0为完全平衡)")
    
    if balance_ratio < 0.5:
        print(f"  ⚠️  类别不平衡，建议增加少数类别样本")

def analyze_qa(data):
    """分析问答数据集"""
    print(f"\n=== 数据平台问答数据集 专项分析 ===")
    
    # 分析问题类型
    question_types = []
    for item in data:
        question = item['input']
        if question.endswith('？') or question.endswith('?'):
            question_types.append('疑问句')
        elif '如何' in question or '怎么' in question:
            question_types.append('方法询问')
        elif '是什么' in question or '什么是' in question:
            question_types.append('概念询问')
        else:
            question_types.append('陈述式')
    
    type_counter = Counter(question_types)
    print(f"问题类型分布:")
    for qtype, count in type_counter.items():
        print(f"  {qtype}: {count}次")
    
    # 分析答案质量
    print(f"\n答案质量分析:")
    structured_answers = 0
    detailed_answers = 0
    
    for item in data:
        answer = item['output']
        if '1.' in answer or '2.' in answer or '：' in answer:
            structured_answers += 1
        if len(answer) > 50:
            detailed_answers += 1
    
    print(f"  结构化答案: {structured_answers}个 ({structured_answers/len(data)*100:.1f}%)")
    print(f"  详细答案(>50字): {detailed_answers}个 ({detailed_answers/len(data)*100:.1f}%)")

def analyze_tool_calling(data):
    """分析工具调用数据集"""
    print(f"\n=== 工具调用数据集 专项分析 ===")
    
    # 分析函数调用分布
    function_calls = []
    for item in data:
        output = item['output']
        # 提取函数名
        if '"name"' in output:
            try:
                # 使用正则表达式提取函数名
                match = re.search(r'"name":\s*"([^"]+)"', output)
                if match:
                    function_calls.append(match.group(1))
                else:
                    function_calls.append('未识别')
            except:
                function_calls.append('解析错误')
        else:
            function_calls.append('格式错误')
    
    func_counter = Counter(function_calls)
    print(f"函数调用分布:")
    for func, count in func_counter.items():
        print(f"  {func}: {count}次")
    
    # 检查输出格式
    print(f"\n输出格式检查:")
    valid_format = 0
    for item in data:
        output = item['output']
        if '<tool_call>' in output and '"name"' in output and '"arguments"' in output:
            valid_format += 1
    
    print(f"  有效格式: {valid_format}个 ({valid_format/len(data)*100:.1f}%)")
    
    # 分析输入复杂度
    print(f"\n输入复杂度分析:")
    complex_inputs = 0
    for item in data:
        input_text = item['input']
        if len(input_text) > 50 and ('数据' in input_text or '文件' in input_text):
            complex_inputs += 1
    
    print(f"  复杂输入: {complex_inputs}个 ({complex_inputs/len(data)*100:.1f}%)")

def provide_optimization_suggestions(dataset_key, data):
    """提供优化建议"""
    print(f"\n=== 优化建议 ===")
    
    if dataset_key == "data_extraction":
        # 数据提取优化建议
        outputs = [item['output'] for item in data]
        unique_outputs = len(set(outputs))
        
        print(f"数据提取数据集优化建议:")
        print(f"1. 数据多样性: 当前有{unique_outputs}种不同的数据类型")
        if unique_outputs < 20:
            print(f"   建议: 增加更多数据类型，目标30-50种")
        
        # 检查输入模式多样性
        pattern_words = ['怎么', '如何', '需要', '想要', '进行', '处理']
        pattern_coverage = sum(1 for item in data for word in pattern_words if word in item['input'])
        print(f"2. 输入模式: 覆盖{pattern_coverage}个模式词")
        print(f"   建议: 增加更多表达方式，如'请帮我'、'能否'、'麻烦'等")
        
    elif dataset_key == "question_classifier":
        # 问题分类优化建议
        outputs = [item['output'] for item in data]
        output_counter = Counter(outputs)
        
        print(f"问题分类数据集优化建议:")
        for category, count in output_counter.items():
            print(f"1. {category}: {count}个样本")
        
        min_count = min(output_counter.values())
        max_count = max(output_counter.values())
        if min_count / max_count < 0.7:
            print(f"   建议: 平衡各类别样本数量，最少类别需要增加{max_count - min_count}个样本")
        
    elif dataset_key == "question_type_classifier":
        # 问题类型分类优化建议
        outputs = [item['output'] for item in data]
        output_counter = Counter(outputs)
        
        print(f"问题类型分类数据集优化建议:")
        for qtype, count in output_counter.items():
            print(f"1. {qtype}: {count}个样本")
        
        # 检查边界案例
        boundary_cases = 0
        for item in data:
            if '帮我' in item['input'] or '请' in item['input']:
                boundary_cases += 1
        
        print(f"2. 边界案例: {boundary_cases}个")
        print(f"   建议: 增加更多模糊边界的样本，提高分类准确性")
        
    elif dataset_key == "dp_qa":
        # 问答数据集优化建议
        print(f"数据平台问答数据集优化建议:")
        
        # 检查答案长度分布
        answer_lengths = [len(item['output']) for item in data]
        avg_length = sum(answer_lengths) / len(answer_lengths)
        
        print(f"1. 答案长度: 平均{avg_length:.1f}字")
        if avg_length < 50:
            print(f"   建议: 增加答案详细程度，提供更多背景信息和步骤")
        
        # 检查结构化程度
        structured = sum(1 for item in data if '1.' in item['output'] or '：' in item['output'])
        print(f"2. 结构化答案: {structured}个 ({structured/len(data)*100:.1f}%)")
        if structured / len(data) < 0.5:
            print(f"   建议: 增加结构化答案，使用编号、分点等格式")
            
    elif dataset_key == "tool_data_platform":
        # 工具调用优化建议
        print(f"工具调用数据集优化建议:")
        
        # 检查函数覆盖度
        function_calls = []
        for item in data:
            match = re.search(r'"name":\s*"([^"]+)"', item['output'])
            if match:
                function_calls.append(match.group(1))
        
        func_counter = Counter(function_calls)
        print(f"1. 函数覆盖: {len(func_counter)}个不同函数")
        
        for func, count in func_counter.items():
            if count < 3:
                print(f"   建议: {func} 只有{count}个样本，建议增加到5-10个")
        
        # 检查参数复杂度
        complex_params = 0
        for item in data:
            if '"arguments"' in item['output']:
                # 简单计算参数数量
                param_count = item['output'].count('":')
                if param_count > 3:
                    complex_params += 1
        
        print(f"2. 复杂参数调用: {complex_params}个")
        print(f"   建议: 增加更多复杂参数的样本，覆盖边界情况")

def main():
    """主分析函数"""
    print("=" * 60)
    print("数据集质量分析报告")
    print("=" * 60)
    
    for dataset_key, config in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"分析数据集: {config['name']}")
        print(f"{'='*80}")
        
        # 加载数据集
        data = load_dataset(config['path'])
        if data is None:
            print(f"跳过 {config['name']} - 文件不存在或加载失败")
            continue
        
        # 基础统计分析
        analyze_basic_stats(data, config['name'])
        
        # 专项分析
        if dataset_key == "data_extraction":
            analyze_data_extraction(data)
        elif dataset_key in ["question_classifier", "question_type_classifier"]:
            analyze_classification(data, config['name'])
        elif dataset_key == "dp_qa":
            analyze_qa(data)
        elif dataset_key == "tool_data_platform":
            analyze_tool_calling(data)
        
        # 优化建议
        provide_optimization_suggestions(dataset_key, data)
        
        print("\n" + "-" * 80)
    
    # 总体建议
    print(f"\n{'='*80}")
    print("总体优化建议")
    print(f"{'='*80}")
    
    print("""
    1. 数据集规模建议:
       - 每个任务至少100-500个样本
       - 分类任务确保类别平衡
       - 复杂任务(如工具调用)需要更多样本
    
    2. 数据质量建议:
       - 增加输入表达的多样性
       - 确保输出格式的一致性
       - 添加边界案例和困难样本
    
    3. 训练效果建议:
       - 考虑添加验证集
       - 实施数据增强技术
       - 定期评估和迭代优化
    
    4. 格式规范建议:
       - 统一JSON格式
       - 确保编码一致性(UTF-8)
       - 添加数据版本管理
    """)

if __name__ == "__main__":
    main()
