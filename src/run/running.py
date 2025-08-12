import torch

from src.train.net import Model

from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = ["负向评价", "正向评价"]

print(device)

model = Model().to(device)

model_name = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
# 加载分词器
tokenizer = BertTokenizer.from_pretrained(model_name)


# 将数据进行编码处理
def collate_fn(data):
    sentes = []
    sentes.append(data)
    print("dayin:", sentes)
    # 编码
    data = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=sentes,
        truncation=True,
        padding="max_length",
        max_length=200,
        return_tensors="pt",
        return_length=True
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids, attention_mask, token_type_ids


def test():
    model.load_state_dict(torch.load(r"E:\project\python\slm_model_train\src\train\params\0bert.pt"))
    model.eval()  # 测试模式
    while True:
        data = input("请输入测试数据（输入'q'退出）：")
        if data == "q":
            print("测试结束")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data)
        input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device)
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定：", names[out], "\n")


if __name__ == '__main__':
    test()
