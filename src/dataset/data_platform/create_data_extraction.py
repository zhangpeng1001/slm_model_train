import json


def _augment_sample(target):
    """数据增强：生成样本变体"""
    augmented = []

    # 问题模板变体
    question_templates = [
        f"如何处理{target}数据？",
        f"{target}数据怎么入库？",
        f"我需要对{target}数据进行处理",
        f"{target}数据的质量检查怎么做？",
        f"请帮我处理{target}数据",
        f"{target}数据如何发布服务？",
        f"我有{target}数据需要治理",
        f"{target}数据的标准化流程",
        f"如何管理{target}数据？",
        f"{target}数据处理完成后如何使用？"
    ]

    for template in question_templates:
        augmented.append((template, target))

    return augmented


def prepare_training_data():
    """准备训练数据"""

    # 训练样本：(问题文本, 提取目标)
    training_samples = [
        ("我有一批西安市地类图斑的数据，怎么进行发服务", "西安市地类图斑"),
        ("请帮我把实景三维模型成果数据进行治理", "实景三维模型成果"),
        ("我已经上传了单波段浮点投影的数据，现在想进行入库", "单波段浮点投影"),
        ("有一些遥感影像数据需要处理", "遥感影像"),
        ("DEM高程数据怎么进行质量检查", "DEM高程"),
        ("矢量地形图数据如何入库", "矢量地形图"),
        ("我的点云数据处理完了，如何发布服务", "点云"),
        ("正射影像数据需要进行坐标转换", "正射影像"),
        ("地理信息数据库的数据怎么备份", "地理信息数据库"),
        ("卫星遥感数据的预处理流程是什么", "卫星遥感"),
        ("无人机航拍数据如何进行几何纠正", "无人机航拍"),
        ("激光雷达数据的格式转换", "激光雷达"),
        ("地籍调查数据需要标准化处理", "地籍调查"),
        ("土地利用现状数据怎么更新", "土地利用现状"),
        ("基础地理信息数据如何管理", "基础地理信息"),
        ("城市三维建模数据的质量控制", "城市三维建模"),
        ("地下管线探测数据处理", "地下管线探测"),
        ("倾斜摄影测量数据建模", "倾斜摄影测量"),
        ("地质勘探数据入库流程", "地质勘探"),
        ("水文监测数据的时序分析", "水文监测"),
        ("气象观测数据处理", "气象观测"),
        ("土壤调查数据分析", "土壤调查"),
        ("森林资源调查数据", "森林资源调查"),
        ("海洋测绘数据处理", "海洋测绘"),
        ("地震监测数据分析", "地震监测"),
        ("环境监测数据入库", "环境监测"),
        ("交通流量数据统计", "交通流量"),
        ("人口普查数据处理", "人口普查"),
        ("经济统计数据分析", "经济统计"),
        ("建筑信息模型数据", "建筑信息模型"),
        ("管网拓扑数据维护", "管网拓扑"),
        ("电力设施数据更新", "电力设施"),
        ("通信基站数据管理", "通信基站"),
        ("道路网络数据处理", "道路网络"),
        ("地名地址数据标准化", "地名地址"),
        ("行政区划数据更新", "行政区划"),
        ("地价监测数据分析", "地价监测"),
        ("规划用地数据管理", "规划用地")
    ]

    data_list = []

    for text, target in training_samples:
        data_dict = {"input": text, "output": target}
        data_list.append(data_dict)

        # 数据增强：创建变体
        augmented_samples = _augment_sample(target)
        for aug_text, aug_target in augmented_samples:
            data_dict = {"input": aug_text, "output": aug_target}
            data_list.append(data_dict)

    print(f"准备了 {len(data_list)} 个训练样本")
    return data_list


print(json.dumps(prepare_training_data(), ensure_ascii=False, indent=2))
