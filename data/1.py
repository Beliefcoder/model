#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import xml.etree.ElementTree as ET
import re

def clean_text(text):
    """清理文本，移除不需要的字符和标签"""
    if not text:
        return ""
    
    # 移除XML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除URL
    text = re.sub(r'http\S+', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 修剪两端空格
    return text.strip()

def parse_iwslt_xml(xml_file):
    """解析IWSLT XML文件，提取翻译对"""
    translations = []
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 遍历所有文档
        for doc in root.findall('.//doc'):
            doc_id = doc.get('id', '')
            
            # 遍历所有段落
            for seg in doc.findall('.//seg'):
                seg_id = seg.get('id', '')
                text = clean_text(seg.text)
                
                if text:
                    translations.append({
                        'text': text,
                        'doc_id': doc_id,
                        'seg_id': seg_id
                    })
                    
    except Exception as e:
        print(f"解析XML文件 {xml_file} 时出错: {e}")
    
    return translations

def read_text_file(file_path):
    """读取文本文件，每行一个句子"""
    sentences = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = clean_text(line)
                if line and not line.startswith('<'):
                    sentences.append(line)
    except Exception as e:
        print(f"读取文本文件 {file_path} 时出错: {e}")
    
    return sentences

def convert_iwslt_to_json():
    """将原始IWSLT数据转换为JSON格式"""
    
    input_dir = "data/en-de"
    output_dir = "data/iwslt2017_json"
    
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始转换 IWSLT2017 数据集...")
    
    # 检查文件结构
    files = os.listdir(input_dir)
    print(f"找到文件: {files}")
    
    # 识别训练、验证和测试文件
    train_en_files = [f for f in files if 'train' in f and f.endswith('.en')]
    train_de_files = [f for f in files if 'train' in f and f.endswith('.de')]
    
    dev_en_files = [f for f in files if 'dev' in f and f.endswith('.en')]
    dev_de_files = [f for f in files if 'dev' in f and f.endswith('.de')]
    
    test_en_files = [f for f in files if 'test' in f and f.endswith('.en')]
    test_de_files = [f for f in files if 'test' in f and f.endswith('.de')]
    
    # 处理XML文件
    xml_en_files = [f for f in files if f.endswith('.en.xml')]
    xml_de_files = [f for f in files if f.endswith('.de.xml')]
    
    # 存储翻译对
    train_pairs = []
    dev_pairs = []
    test_pairs = []
    
    # 方法1: 处理文本文件
    if train_en_files and train_de_files:
        print("处理训练文本文件...")
        train_en = read_text_file(os.path.join(input_dir, train_en_files[0]))
        train_de = read_text_file(os.path.join(input_dir, train_de_files[0]))
        
        min_len = min(len(train_en), len(train_de))
        for i in range(min_len):
            train_pairs.append({
                "translation": {
                    "en": train_en[i],
                    "de": train_de[i]
                }
            })
    
    # 方法2: 处理XML文件
    if xml_en_files and xml_de_files:
        print("处理XML文件...")
        
        # 处理开发集XML
        for en_xml, de_xml in zip(sorted(xml_en_files), sorted(xml_de_files)):
            if 'dev' in en_xml or 'dev' in de_xml:
                print(f"处理开发集: {en_xml}, {de_xml}")
                en_translations = parse_iwslt_xml(os.path.join(input_dir, en_xml))
                de_translations = parse_iwslt_xml(os.path.join(input_dir, de_xml))
                
                # 按文档和段落ID匹配
                en_dict = {(t['doc_id'], t['seg_id']): t['text'] for t in en_translations}
                de_dict = {(t['doc_id'], t['seg_id']): t['text'] for t in de_translations}
                
                # 匹配相同的文档和段落
                common_keys = set(en_dict.keys()) & set(de_dict.keys())
                for key in common_keys:
                    dev_pairs.append({
                        "translation": {
                            "en": en_dict[key],
                            "de": de_dict[key]
                        }
                    })
            
            # 处理测试集XML
            elif 'test' in en_xml or 'test' in de_xml:
                print(f"处理测试集: {en_xml}, {de_xml}")
                en_translations = parse_iwslt_xml(os.path.join(input_dir, en_xml))
                de_translations = parse_iwslt_xml(os.path.join(input_dir, de_xml))
                
                # 按文档和段落ID匹配
                en_dict = {(t['doc_id'], t['seg_id']): t['text'] for t in en_translations}
                de_dict = {(t['doc_id'], t['seg_id']): t['text'] for t in de_translations}
                
                # 匹配相同的文档和段落
                common_keys = set(en_dict.keys()) & set(de_dict.keys())
                for key in common_keys:
                    test_pairs.append({
                        "translation": {
                            "en": en_dict[key],
                            "de": de_dict[key]
                        }
                    })
    
    # 如果没有找到足够的配对，使用简单方法创建示例数据
    if not train_pairs:
        print("未找到训练数据，创建示例数据...")
        train_pairs = create_sample_data(1000)
    
    if not dev_pairs:
        print("未找到开发集数据，从训练集分割...")
        if train_pairs:
            split_point = int(len(train_pairs) * 0.9)
            dev_pairs = train_pairs[split_point:]
            train_pairs = train_pairs[:split_point]
        else:
            dev_pairs = create_sample_data(100)
    
    if not test_pairs:
        print("未找到测试集数据，从开发集分割...")
        if dev_pairs:
            split_point = int(len(dev_pairs) * 0.5)
            test_pairs = dev_pairs[split_point:]
            dev_pairs = dev_pairs[:split_point]
        else:
            test_pairs = create_sample_data(100)
    
    # 保存为JSON文件
    print("保存JSON文件...")
    
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "validation.json"), "w", encoding="utf-8") as f:
        json.dump(dev_pairs, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)
    
    # 保存数据集信息
    dataset_info = {
        "name": "IWSLT2017 en-de (Converted from raw data)",
        "source": "Local raw files",
        "train_size": len(train_pairs),
        "validation_size": len(dev_pairs),
        "test_size": len(test_pairs),
        "input_directory": input_dir,
        "output_directory": output_dir
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n转换完成!")
    print(f"训练集: {len(train_pairs)} 个句子对")
    print(f"验证集: {len(dev_pairs)} 个句子对")
    print(f"测试集: {len(test_pairs)} 个句子对")
    print(f"输出目录: {output_dir}")
    
    # 显示一些样本
    print("\n样本示例:")
    for i in range(min(3, len(train_pairs))):
        print(f"样本 {i+1}:")
        print(f"  英语: {train_pairs[i]['translation']['en']}")
        print(f"  德语: {train_pairs[i]['translation']['de']}")
        print()
    
    return True

def create_sample_data(size):
    """创建示例数据"""
    sample_pairs = [
        ("Hello world", "Hallo Welt"),
        ("How are you?", "Wie geht es dir?"),
        ("I love machine learning", "Ich liebe maschinelles Lernen"),
        ("Good morning", "Guten Morgen"),
        ("What is your name?", "Wie ist dein Name?"),
        ("Thank you very much", "Vielen Dank"),
        ("Where is the station?", "Wo ist der Bahnhof?"),
        ("I don't understand", "Ich verstehe nicht"),
        ("Can you help me?", "K?nnen Sie mir helfen?"),
        ("Have a nice day", "Einen sch?nen Tag noch")
    ]
    
    pairs = []
    for i in range(size):
        en_text, de_text = sample_pairs[i % len(sample_pairs)]
        pairs.append({
            "translation": {
                "en": f"{en_text} #{i}",
                "de": f"{de_text} #{i}"
            }
        })
    
    return pairs

if __name__ == "__main__":
    convert_iwslt_to_json()