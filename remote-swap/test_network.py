# remote_swap/test_network.py
import torch
from remote_swap.client import RemoteKVClient

if __name__ == "__main__":
    client = RemoteKVClient("127.0.0.1",12345)
    test_tensor = torch.randn((32,2,16,32,128))
    bid = 1001
    client.put_block(bid, test_tensor)
    print("上传成功")
    load_tensor = client.get_block(bid)
    print("下载成功，形状", load_tensor.shape)
    print("数值相等：", torch.allclose(test_tensor, load_tensor))