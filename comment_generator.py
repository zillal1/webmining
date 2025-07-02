from WM.model import LLM
import json
from zhipuai import ZhipuAI


def Generate_Comment(phrase: str, max_iter=3,
                     api_key='b22e534e774d45a7be57cb0bee1bc89e.CICzXPz4EpwnW3uO'):
    client = ZhipuAI(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": f'{LLM.build_prompt_comment(phrase)}'

        }
    ]
    # print(messages)
    try:
        for _ in range(max_iter):

            response = client.chat.completions.create(
                model="glm-4-plus",  # 请填写您要调用的模型名称
                messages=messages,
                tool_choice="auto",
            )
            # print((dict(response.choices[0])['message']).content)
            a = (dict(response.choices[0])['message']).content
            if a and len(a) > 0:
                return a

        return 'Fail to execute'
    except SyntaxError:
        print('Fail! Got Output:', (dict(response.choices[0])['message']).content)
        return 'Fail to execute'


def generate_ultimate_comment(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = str(json.load(f))
    # print(data)
    result = Generate_Comment(data)
    print(result)
    return result


if __name__ == '__main__':
    jsonfile = 'data/logs/sentiment/sentiment_result_20250701_2341.json'
    generate_ultimate_comment(jsonfile)
