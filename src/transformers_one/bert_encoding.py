# bert 编码

from transformers import BertTokenizer

# 加载字典和分词工具
model_name = r"E:\project\python\slm_model_train\src\transformers_one\model\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"

tokenizer = BertTokenizer.from_pretrained(model_name)

sents = [
    "宝贝超棒！质量杠杠的，完全符合描述，用起来超顺手，必须好评",
    "这商品和图片差距有点大，实物质感一般，不太满意，希望商家改进,物流速度太慢了，等了好久才收到货，好在产品还行，不然真要给差评,东西收到，质量不咋地，做工粗糙，细节处理不到位，不建议大家购买",
    "买的衣服已收到，面料舒适，款式时尚，尺码也合适，很喜欢，推荐购买"
]
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,
    truncation=True,  # 当句子长度大于max_length时，截断
    padding="max_length",  # 一律补零到max_length的长度
    max_length=30,
    return_tensors=None,  # 可取值tf,pt,np,默认为返回list
    return_attention_mask=True,  # 返回attention_mask
    return_special_tokens_mask=True,  # 返回special_tokens_mask
    # return_offsets_mapping=True,#返回offsets_mapping，标识每个词的起止位置，这个参数只能BertTokenizerFast使用
    return_length=True,  # 返回length标识长度
)
print("*" * 30)
print(out)
print("*" * 30)

print("=" * 30)
for k, v in out.items():
    print(k, ":", v)

print("=" * 30)
print(tokenizer.decode(out["input_ids"][0]), tokenizer.decode(out["input_ids"][1]))

