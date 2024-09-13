from openai import OpenAI
from key import openai_key, wit_key

client = OpenAI(
    api_key=openai_key,
)

def get_completion(prompt, model="gpt-4-turbo", temperature=0):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model,
    messages=messages,
    temperature=temperature)
    return response.choices[0].message.content


def q_with_data(question, data):

    prompt = f"""
    Your task is to answer with following question using the data delimited by <>.

    {question}

    <{data}>
    """

    return get_completion(prompt)


def samples():
    #response = get_completion('What are the best jobs going to be in 20 years?')
    #print(f'{response}\n')

    #response = get_completion('Write Python code to compute the Fibonacci sequence')
    #print(f'{response}\n')

    response = get_completion('Write a short essay about the book Harry Potter')
    print(f'{response}\n')

    response = get_completion('Create ASCII art of some fun characters')
    #with open('art.txt', 'w') as f:
    #    f.write(response)
    print(f'{response}\n')


if __name__ == "__main__":
    samples()
