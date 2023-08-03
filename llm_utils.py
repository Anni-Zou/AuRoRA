import time
import openai


# Sentence Generator (Decoder) for GPT-3 ...
def decoder_for_gpt3(input, max_length, engine, temperature=0):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    if engine == "gpt-3.5-turbo":
        time.sleep(20)
        response  = openai.ChatCompletion.create(
            model=engine,
            messages=[
                #{"role": "system", "content": "You need to answer commonsense questions."},
                {"role": "user", "content": input}
            ],
            max_tokens=max_length,
            temperature=temperature,
            stop=None
        )
        response = response["choices"][0]["message"]["content"]

    else:
        time.sleep(1)
        response = openai.Completion.create(
            model=engine,
            prompt=input,
            max_tokens=max_length,
            stop=None,
            temperature=temperature
        )
        response = response["choices"][0]["text"]
    return response

def decoder_for_gpt3_consistency(input, max_length, engine, temp=0.7, n=5):
    # GPT-3 API allows each users execute the API within 60 times in a minute ...
    if engine == "gpt-3.5-turbo":
        time.sleep(20)
        responses = openai.ChatCompletion.create(
            model=engine,
            messages=[
                {"role": "user", "content": input}
            ],
            max_tokens=max_length,
            temperature=temp,
            top_p=1,
            n=5,
            stop=["\n"],
        )
        responses = [responses["choices"][i]["message"]["content"] for i in range(n)]
    else:
        time.sleep(1)
        responses = openai.Completion.create(
            model=engine,
            prompt=input,
            max_tokens=max_length,
            temperature=temp,
            stop=["\n"],
            n=5,
            logprobs=5,
            top_p=1,
        )
        responses = [responses["choices"][i]["text"] for i in range(n)]

    return responses