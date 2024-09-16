from openai import OpenAI
from key import openai_key

class ChatBot:

    def __init__(self) -> None:
        self.__client = OpenAI(api_key=openai_key)

        self.__context = [ {'role':'system', 'content':""" \
                    You are an OrderBot, an automated service to collect orders for a pizza restaurant. \
                    You first greet the customer, then collect the order and then ask if it's a pickup or delivery. \
                    You wait to collect the entire order, then summarize it and check for a final time if the customer wants to add anything else. \
                    If it's a delivery, you ask for an address. \
                    Finally you collect the payment.\
                    Make sure to clarify all options, extras and sizes to uniquely identify the item from the menu.\
                    You respond in a short, very conversational friendly style. \
                    Don't allow orders for things not on the menu. \
                    The menu includes \
                    pepperoni pizza  12.95, 10.00, 7.00 \
                    cheese pizza   10.95, 9.25, 6.50 \
                    eggplant pizza   11.95, 9.75, 6.75 \
                    fries 4.50, 3.50 \
                    greek salad 7.25 \
                    Toppings: \
                    extra cheese 2.00, \
                    mushrooms 1.50 \
                    sausage 3.00 \
                    canadian bacon 3.50 \
                    AI sauce 1.50 \
                    peppers 1.00 \
                    Drinks: \
                    coke 3.00, 2.00, 1.00 \
                    sprite 3.00, 2.00, 1.00 \
                    bottled water 5.00 \
                    """} ]  # accumulate messages
        
    def get_completion_from_messages(self, messages, model="gpt-4-turbo", temperature=0):
        response = self.__client.chat.completions.create(model=model, messages=messages, temperature=temperature)
        return response.choices[0].message.content
    
    def respond(self, prompt):
        self.__context.append({'role':'user', 'content':f"{prompt}"})
        response = self.get_completion_from_messages(self.__context) 
        self.__context.append({'role':'assistant', 'content':f"{response}"})
        return response

if __name__ == "__main__":
    chatbot = ChatBot()

    human_prompt = ""
    while human_prompt != 'done':
        response = chatbot.respond(human_prompt)
        human_prompt = input(f"\n{response}\n\n")
        