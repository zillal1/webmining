from zhipuai import ZhipuAI
import json


def build_prompt(phrase):
    """
    This function creates the specific message (prompt) to be sent to the LLM.
    """
    # The message content starts here
    prompt = f'''You are an expert in structuring product review data. Your task is to assign the following phrase from a customer review to the most appropriate category (aspect) from the list below.

    The aspects and their definitions are:
    1. quality: About build quality, material, design, durability, workmanship, or appearance
    2. performance: About device speed, stability, system fluidity, responsiveness, or lag
    3. display: About the screen, display effect, color, resolution, or brightness
    4. price: About cost, price, value for money, affordability, or worth
    5. service: About customer service, after-sales, support, repair, or warranty
    6. Component: About each component of the object such as battery, chip
    7. others: If the phrase is too general (such as 'love', 'like', 'ok', 'nice', 'good') or does not relate clearly to any of the above, classify as 'others'.

    Please output in strict JSON format as below, and do not add any extra explanation:
    {{"input_phrase":"{phrase}","aspect":"assigned_aspect_name","reason":"One-sentence explanation for your assignment"}}
    
    Please notice the start and the ending of your answer should be {{ and }} respectively.
    
    Also, don't do too much inference, if some phrase is strange, you just need to classify it as "others"
    
    Example1:
    Question:Finish
    Answer:{{"input_phrase":"Finish","aspect":"quality","reason":"finish usually shows a running speed of a device, so it should belong to quality part"}}
    
    Example2:
    Question:Garbage
    Answer:{{"input_phrase":"Garbage","aspect":"others","reason":"Garbage has less relationship with the other 6 categories so it shouold belong to others"}}
    
     Example2:
    Question:Garbage
    Answer:{{"input_phrase":"Chip Chicken","aspect":"others","reason":"Even though chip can be a component, but chip chicken is a strange phrase that people never see, so it should belong to 'others'"}}



    The phrase to classify: "{phrase}"'''

    # =================== End of Prompt ======================================
    return prompt.strip()


def build_prompt_comment(prompt):
    prompt = f"""
    You will receive a JSON array summarizing various product aspects—such as quality, component, performance, display, price, service. Each entry contains:
    
    aspect: the name of the aspect
    n_mention: number of times this aspect is mentioned
    avg_sentiment: average sentiment score for this aspect (negative means negative opinion, positive means positive, 0 is neutral)
    pos_example: a representative positive comment for this aspect (may be empty), if it's none, it means that most users give a neutral attitude in this aspect.
    neg_example: a representative negative comment for this aspect (may be empty), if it's none, it means that most users give a neutral attitude in this aspect
    
    
    Your task is to analyze and review the product as follows:
    
    Output Format:
    
    1.Overall Summary:
    Start with a concise summary describing the general user impression of the product.
    
    2.Aspect-by-Aspect Analysis:
    For each aspect in the input (in the order provided), produce a short section in this format:
    
    Aspect Name (aspect):
        Mentions: You need to use star to show the frequency of mentioning, for example, if this aspect appears very frequently, marks 5stars, otherwise marks 1-4 stars.
        Average Sentiment: avg_sentiment (positive/negative/neutral as appropriate), you also need to use star to describe it, 5 it the highest while 1 is the lowest
        Main strengths and weaknesses: in your own words, summarize the key pros and cons for this aspect based on the samples and sentiment.
        Briefly quote or paraphrase the most telling points from the pos_example and neg_example (if they are non-empty). Do not just copy; instead, summarize their core meaning.
        Final Recommendation:
        Offer a concise, actionable summary: suggest what could be improved, which aspects are strengths, and your overall verdict for potential buyers.
    
    Rules:
    
    Make sure your language is clear, logical, and natural for a human reader.
    Avoid only restating numbers; focus on interpretation, insight, and readability.
    Do not skip any aspect present in the input.
    Do not add any aspect that DOES NOT appear in the given json file.
    
    
    The input json is:
    {prompt} 
    """
    return prompt.strip()


def Add_New_Member(phrase: str, class_list: dict, max_iter=3,
                   api_key='b22e534e774d45a7be57cb0bee1bc89e.CICzXPz4EpwnW3uO'):
    client = ZhipuAI(api_key=api_key)

    messages = [
        {
            "role": "user",
            "content": f'{build_prompt(phrase)}'

        }
    ]
    try:
        for _ in range(max_iter):

            response = client.chat.completions.create(
                model="glm-4-plus",  # 请填写您要调用的模型名称
                messages=messages,
                tool_choice="auto",
            )
            print((dict(response.choices[0])['message']).content)
            a = eval((dict(response.choices[0])['message']).content)
            if isinstance(a, dict) and a['aspect'] in class_list.keys():
                return a['aspect']

        return None
    except SyntaxError:
        print('Fail! Got Output:', (dict(response.choices[0])['message']).content)
        return None


if __name__ == '__main__':
    ASPECT_MAP = {
        "quality": ["quality", "material", "build", "solid", "design", "durable", "finish"],
        "Component": ["chip", "battery"],
        "performance": ["performance", "speed", "lag", "slow", "fast", "responsive"],
        "display": ["screen", "display", "resolution", "brightness", "touch"],
        "price": ["price", "value", "cost", "cheap", "expensive", "worth"],
        "service": ["service", "support", "warranty", "helpful", "replacement", "customer service"],
        "others": ['love', 'like', 'nice', 'great']
    }
    print(Add_New_Member('computer chicken', ASPECT_MAP))
