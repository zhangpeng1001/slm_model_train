# 上游任务 - 预训练模型
from transformers import BertModel
# 下游任务 - pytorch
import torch

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

# 将模型放入到设备中，参与后期推理
pretained = BertModel.from_pretrained(model_name).to(device)

print(pretained)


# 定制化输入的时候，需要修改embeddings.word_embeddings， -----模型微调时，使用
# print(pretained.embeddings)
# print(pretained.embeddings.word_embeddings)

# 输入的数据经过 bert模型后，输出为768（pretained的输出部分 Linear(in_features=768, out_features=768, bias=True)）

# 定义下游任务，将主干网络提取的特征（768），进行文本分类
class Model(torch.nn.Module):
    # 模型结构设计
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # 推理过程
        # 上游任务，bert模型不参与训练，bert只是将文本提取成768个特征向量
        with torch.no_grad():
            out = pretained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 下游任务，参与训练
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out
