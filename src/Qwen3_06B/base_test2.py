from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = r'E:\project\llm\model-data\base-models\Qwen3-0.6B'


class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response


# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot(model_name)

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "草莓（strawberries）中有多少个字母 r？"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "那么，“blueberries” 这个单词里有多少个字母 “r” 呢？/no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}")
    print("----------------------")

    # Third input with /think
    user_input_3 = "真的吗？/think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
