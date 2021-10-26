import torch
from sys import platform
from torch.utils.data import DataLoader
from data import MyDataset, load_embeddings
from model import SiaGRU
from utils import test

def main(test_file, vocab_file, embeddings_file, pretrained_file, max_length=30, batch_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for testing ", 20 * "=")
    if platform == "linux" or platform == "linux2":#判断平台
        checkpoint = torch.load(pretrained_file)#加载模型
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    #加载预训练字向量  只是用来初始化
    embeddings = load_embeddings(embeddings_file)
    print("\t* Loading test data...")
    #加载测试数据
    test_data = MyDataset(test_file, vocab_file, max_length)
    #生成数据迭代器
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    print("\t* Building model...")
    #定义模型
    model = SiaGRU(embeddings,max_len=max_length, device=device).to(device)
    #加载模型参数
    model.load_state_dict(checkpoint["model"])
    print(20 * "=", " Testing SiaGRU model on device: {} ".format(device), 20 * "=")
    #调用test函数完成评估
    batch_time, total_time, accuracy, auc = test(model, test_loader)
    print("\n-> Average batch processing time: {:.4f}s, total test time: {:.4f}s, accuracy: {:.4f}%, auc: {:.4f}\n".format(batch_time, total_time, (accuracy*100), auc))


if __name__ == "__main__":
    main("data/test.csv", "data/vocab.txt", "data/token_vec_300.bin", "models/best.pth.tar")