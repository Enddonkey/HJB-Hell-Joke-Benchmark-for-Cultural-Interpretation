import pandas as pd
import openai
import json
import os
import re
import argparse
import concurrent.futures # 导入并发模块

def load_api_config(config_file='../openai_key.txt'):
    """从配置文件加载 API 密钥和基础 URL"""
    api_key = None
    base_url = None
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY'):
                        # 支持多种分隔符: 中文冒号、英文冒号、等号
                        if '：' in line:
                            api_key = line.split('：', 1)[1].strip()
                        elif ':' in line:
                            api_key = line.split(':', 1)[1].strip()
                        elif '=' in line:
                            api_key = line.split('=', 1)[1].strip()
                    elif line.startswith('OPENAI_BASE_URL'):
                        # 支持逗号或冒号分隔
                        if ',' in line:
                            base_url = line.split(',', 1)[1].strip()
                        elif '：' in line:
                            base_url = line.split('：', 1)[1].strip()
                        elif ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) > 1:
                                base_url = parts[1].strip()
                                if not base_url.startswith('http'):
                                    base_url = ':' + base_url
        except Exception as e:
            print(f"Warning: 读取配置文件 {config_file} 时出错: {e}")
    
    # 如果配置文件中没有，则从环境变量获取
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL")
    
    return api_key, base_url

# 全局变量，将在 main 函数中初始化
client = None

def analyze_joke_with_openai(joke_content, model_name, temperature=0.7, max_tokens=5000):
    """
    使用 OpenAI 兼容 API 分析笑话内容。
    
    参数:
        joke_content: 笑话内容
        model_name: 使用的模型名称
        temperature: 温度参数，控制输出的随机性 (默认: 0.7)
        max_tokens: 最大token数量 (默认: 5000)
    
    返回:
        (background, logic) 元组，包含背景溯源和幽默逻辑
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """你现在是一名冒犯性笑话审查员，下面我将会给你一个笑话，完成以下任务
            1. 背景溯源：请你分析 "这个笑话的核心梗关联了什么真实历史/社会事件/文化典故？"
            2. 幽默逻辑：这个笑话的笑点是如何产生的？请说明其核心逻辑

            请严格按照以下 JSON 格式返回，不要添加任何额外的解释或说明：
            {
            "background": "相关历史事件/文化典故",
            "logic": "具体的逻辑"
            }
        """ },
                {"role": "user", "content": f"笑话内容：\n{joke_content}"}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 检查响应是否为字符串，这通常表示API返回了非预期的错误信息
        if isinstance(response, str):
            print(f"API returned a string response instead of an object. Raw response: {response}")
            return "API Error: Unexpected string response", "API Error: Unexpected string response"

        if response.choices and response.choices[0].message:
            text_response = response.choices[0].message.content
            # 使用正则表达式从回复中提取JSON块
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(0)
                try:
                    # 使用 strict=False 来允许字符串中的控制字符（如换行符）
                    data = json.loads(json_string, strict=False)
                    background = data.get("background", "N/A")
                    logic = data.get("logic", "N/A")
                    return background, logic
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode extracted JSON. Extracted content: {json_string}")
                    return "API Response Error: Invalid Extracted JSON", "API Response Error: Invalid Extracted JSON"
            else:
                print(f"Warning: No JSON object found in API response. Raw content: {text_response}")
                return "API Response Error: No JSON found", "API Response Error: No JSON found"
        else:
            print(f"Warning: No valid response from API for joke: {joke_content}")
            return "API Response Error: No valid message", "API Response Error: No valid message"
    except Exception as e:
        print(f"Error analyzing joke '{joke_content}': {e}")
        return f"API Error: {e}", f"API Error: {e}"

def main():
    global client
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='使用 OpenAI API 分析笑话内容',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 使用默认参数
  python analyze_jokes.py
  
  # 指定输入输出文件
  python analyze_jokes.py --input data/jokes.csv --output results/analyzed.jsonl
  
  # 指定模型和并发数
  python analyze_jokes.py --model gpt-4 --workers 20
  
  # 指定配置文件
  python analyze_jokes.py --config ../openai_key.txt
  
  # 完整参数示例
  python analyze_jokes.py --input data/jokes.csv --output results/output.jsonl --model qwen-max --workers 15 --temperature 0.5 --max-tokens 3000
        '''
    )
    parser.add_argument('--input', type=str, default='./data/offensive - Sheet1.csv',
                        help='输入 CSV 文件路径 (默认: ./data/offensive - Sheet1.csv)')
    parser.add_argument('--output', type=str, default='./qwen/analyzed_jokes.jsonl',
                        help='输出 JSONL 文件路径 (默认: ./qwen/analyzed_jokes.jsonl)')
    parser.add_argument('--model', type=str, default='qwen-max',
                        help='使用的模型名称 (默认: qwen-max)')
    parser.add_argument('--workers', type=int, default=10,
                        help='并发处理的 worker 数量 (默认: 10)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='模型温度参数，范围 0.0-2.0 (默认: 0.7)')
    parser.add_argument('--max-tokens', type=int, default=5000,
                        help='最大 token 数量 (默认: 5000)')
    parser.add_argument('--config', type=str, default='../openai_key.txt',
                        help='API 配置文件路径 (默认: ../openai_key.txt)')
    
    args = parser.parse_args()
    
    # 初始化 OpenAI 客户端
    api_key, base_url = load_api_config(args.config)
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    input_csv_path = args.input
    output_jsonl_path = args.output
    model_name = args.model
    max_workers = args.workers

    if not os.path.exists(input_csv_path):
        print(f"Error: Input CSV file not found at {input_csv_path}")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_jsonl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv_path)
    
    if 'id' not in df.columns or 'content' not in df.columns:
        print("Error: CSV must contain 'id' and 'content' columns.")
        return

    print("="*60)
    print("笑话分析工具 - 配置信息")
    print("="*60)
    print(f"  配置文件: {args.config}")
    print(f"  API Base URL: {base_url}")
    print(f"  输入文件: {input_csv_path}")
    print(f"  输出文件: {output_jsonl_path}")
    print(f"  模型: {model_name}")
    print(f"  并发数: {max_workers}")
    print(f"  温度: {args.temperature}")
    print(f"  最大 Tokens: {args.max_tokens}")
    print(f"  待处理笑话数量: {len(df)}")
    print("="*60)
    print()

    results = []
    # 使用 ThreadPoolExecutor 实现多worker并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备要提交给执行器的任务列表
        future_to_joke = {
            executor.submit(analyze_joke_with_openai, row['content'], model_name, args.temperature, args.max_tokens): row['id'] 
            for index, row in df.iterrows()
        }
        
        completed = 0
        total = len(future_to_joke)
        
        for future in concurrent.futures.as_completed(future_to_joke):
            joke_id = future_to_joke[future]
            try:
                background, logic = future.result()
                joke_content = df[df['id'] == joke_id]['content'].iloc[0] # 获取原始笑话内容
                results.append({
                    "id": joke_id,
                    "content": joke_content,
                    "Background": background,
                    "Logic": logic
                })
                completed += 1
                print(f"[{completed}/{total}] 完成分析笑话 ID: {joke_id}")
            except Exception as exc:
                print(f"笑话 ID {joke_id} 处理异常: {exc}")
                joke_content = df[df['id'] == joke_id]['content'].iloc[0] # 获取原始笑话内容
                results.append({
                    "id": joke_id,
                    "content": joke_content,
                    "Background": f"Error: {exc}",
                    "Logic": f"Error: {exc}"
                })
                completed += 1

    # 按照ID排序结果
    results.sort(key=lambda x: x['id'])

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print()
    print("="*60)
    print(f"分析完成！结果已保存到: {output_jsonl_path}")
    print(f"成功处理: {len([r for r in results if not r['Background'].startswith('Error')])}/{total}")
    print(f"处理失败: {len([r for r in results if r['Background'].startswith('Error')])}/{total}")
    print("="*60)

if __name__ == "__main__":
    main()
