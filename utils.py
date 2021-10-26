import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score



def correct_predictions(preds, targets):
    _, out_classes = preds.max(dim=1)#取出预测概率1轴的最大值
    correct = (out_classes == targets).sum()#跟真实值比较是否相等
    return correct.item()

def train(model, dataloader, optimizer, criterion, max_gradient_norm):
    model.train()#设为训练模式
    device = model.device
    epoch_start = time.time()#起始时间
    batch_time_avg,running_loss,correct_preds = 0.0,0.0,0#时间累加器，总损失累加器，正确样本数累加器
    tqdm_batch_iterator=tqdm(dataloader)
    for batch_index, (q, _, h, _, label) in enumerate(tqdm_batch_iterator):#遍历训练数据批次
        batch_start = time.time()#一个批次的起始时间
        q1,q2,labels= q.to(device),h.to(device),label.to(device)#句子1 句子2，标签放到设备中
        optimizer.zero_grad()#导数清零
        logits, probs = model(q1, q2)#句子1和句子2送入模型计算输出概率
        loss = criterion(logits, labels)#计算损失
        loss.backward()#反向传播求导
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)#梯度截断
        optimizer.step()#参数更新
        batch_time_avg += time.time() - batch_start#计算用时
        running_loss += loss.item()#累加损失值
        correct_preds += correct_predictions(probs, labels)#计算准确率并累加
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)#输出训练信息
    epoch_time = time.time() - epoch_start#一个轮次用时
    epoch_loss = running_loss / len(dataloader)#一个轮次平均
    epoch_accuracy = correct_preds / len(dataloader.dataset)#一个轮次准确率
    return epoch_time, epoch_loss, epoch_accuracy

def validate(model, dataloader, criterion):
    model.eval()#设置模型为验证模式
    device = model.device
    epoch_start = time.time()
    running_loss,running_accuracy = 0.0, 0.0#损失累加器，正确样本数累加器
    all_prob ,all_labels = [],[]#保存所有批次概率和标签
    with torch.no_grad():#验证阶段不求导
        for (q, _, h, _, label) in dataloader:
            q1,q2,labels = q.to(device),h.to(device),label.to(device)#确定设备
            logits, probs = model(q1, q2)#计算模型输出值
            loss = criterion(logits, labels)#计算损失
            running_loss += loss.item()#累加损失
            running_accuracy += correct_predictions(probs, labels)#累加正确个数
            all_prob.extend(probs[:,1].cpu().numpy())
            all_labels.extend(label)
    epoch_time = time.time() - epoch_start#总耗时
    epoch_loss = running_loss / len(dataloader)#平均损失
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))#准确率
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob)


def test(model, dataloader):
    model.eval()#设置为验证模式
    device = model.device
    time_start = time.time()
    batch_time,accuracy = 0.0,0.0
    all_prob, all_labels = [], []  # 保存所有批次概率和标签
    with torch.no_grad():#不求导
        for (q, _, h, _, label) in dataloader:
            batch_start = time.time()
            q1, q2, labels = q.to(device), h.to(device), label.to(device)  # 确定设备
            _, probs = model(q1, q2)#计算模型输出值
            accuracy += correct_predictions(probs, labels)#累加正确个数
            batch_time += time.time() - batch_start
            all_prob.extend(probs[:,1].cpu().numpy())
            all_labels.extend(label)
    batch_time /= len(dataloader)#平均耗时
    total_time = time.time() - time_start#总耗时
    accuracy /= (len(dataloader.dataset))#准确率
    return batch_time, total_time, accuracy, roc_auc_score(all_labels, all_prob)


