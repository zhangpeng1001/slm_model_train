import torch
from torch.utils.data import DataLoader
from net import Model
from myDataset import MyDataset
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100

model_name = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
# 加载分词器
tokenizer = BertTokenizer.from_pretrained(model_name)


# 将数据进行编码处理
def collate_fn(data):
    sentes = [i[0] for i in data]
    label = [i[1] for i in data]
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
    labels = torch.LongTensor(label)
    return input_ids, attention_mask, token_type_ids, labels


# 创建数据集
test_dataset = MyDataset("test")
# 创建DataLoader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=True,  # 数据打乱
    drop_last=True,  # 长度截断
    collate_fn=collate_fn  # 数据编码方法
)

# 开始模型训练
if __name__ == '__main__':
    print(device)
    model = Model().to(device)
    acc = 0
    total = 0
    # optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    # loss_func = torch.nn.CrossEntropyLoss()
    #
    # model.train()
    model.load_state_dict(torch.load("params/0bert.pt"))
    model.eval()  # 测试模式

    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
        # 将数据放到device上
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(
            device), token_type_ids.to(device), labels.to(device)

        # 执行前向计算，得到输出
        out = model(input_ids, attention_mask, token_type_ids)

        out = out.argmax(dim=1)
        acc += (out == labels).sum().item()
        total += len(labels)
    print(acc / total)  # 0.8851351351351351
